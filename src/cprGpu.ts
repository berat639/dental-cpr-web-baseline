import type { ArchResult } from './arch/arch';
import type { Slab } from './cpr';
import type { Volume } from './volume';

const vertex = `#version 300 es
in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }`;

const fragment = `#version 300 es
precision highp float;
precision highp usampler3D;
uniform usampler3D volumeTex;
uniform sampler2D archTex;
uniform sampler2D trackingTex;
uniform bool trackMandible;
uniform ivec3 dimensions;
uniform vec3 spacing;
uniform int columnCount;
uniform int samples;
uniform float thickness;
uniform float windowCenter;
uniform float windowWidth;
uniform int projectionMode;
out vec4 color;

float sampleVolume(vec3 voxel) {
  // Panorama and straight cross-sections both sample an integer source z row.
  // Match the CPU's subvoxel x/y interpolation without quantizing HU to a
  // filterable normalized texture or taking eight trilinear reads per sample.
  if (any(lessThan(voxel, vec3(0.0))) || any(greaterThan(voxel, vec3(dimensions - ivec3(1))))) return -1000.0;
  ivec3 lo = ivec3(floor(voxel));
  ivec2 hi = min(lo.xy + ivec2(1), dimensions.xy - ivec2(1));
  vec2 f = fract(voxel.xy);
  float a = float(texelFetch(volumeTex, lo, 0).r);
  float b = float(texelFetch(volumeTex, ivec3(hi.x, lo.y, lo.z), 0).r);
  float c = float(texelFetch(volumeTex, ivec3(lo.x, hi.y, lo.z), 0).r);
  float d = float(texelFetch(volumeTex, ivec3(hi, lo.z), 0).r);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y) - 32768.0;
}

void main() {
  int column = clamp(int(gl_FragCoord.x), 0, columnCount - 1);
  int row = clamp(int(gl_FragCoord.y), 0, dimensions.z - 1);
  vec4 arch = texelFetch(archTex, ivec2(column, 0), 0);
  // arch.xy is in mm in the image coordinate basis, arch.zw is a unit normal.
  // WebGL fragment coordinates already increase from the bottom of the
  // framebuffer. Keeping the source z index equal to that row matches the
  // conventional canvas CPR convention: high z is drawn at the top.
  vec3 base = vec3(arch.x / spacing.x, arch.y / spacing.y, float(row));
  if (trackMandible) {
    float shift = texelFetch(trackingTex, ivec2(column, row), 0).r;
    base.xy += arch.zw * shift / spacing.xy;
  }
  float value = projectionMode == 1 ? -1.0e20 : 0.0;
  for (int i = -12; i <= 12; ++i) {
    if (abs(i) > samples / 2) continue;
    float offset = samples <= 1 ? 0.0 : float(i) * thickness / float(samples - 1);
    vec3 voxel = base + vec3(arch.z * offset / spacing.x, arch.w * offset / spacing.y, 0.0);
    float current = sampleVolume(voxel);
    if (projectionMode == 1) value = max(value, current); else value += current;
  }
  if (projectionMode == 0) value /= float(samples);
  float grey = clamp((value - (windowCenter - windowWidth * 0.5)) / windowWidth, 0.0, 1.0);
  color = vec4(vec3(grey), 1.0);
}`;

function shader(gl: WebGL2RenderingContext, type: number, source: string) {
  const result = gl.createShader(type)!;
  gl.shaderSource(result, source); gl.compileShader(result);
  if (!gl.getShaderParameter(result, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(result) ?? 'shader compile failed');
  return result;
}

/**
 * WebGL2 curved-planar renderer. The source texture stays on the local GPU;
 * changing window/level or moving the arch uploads only the small arch texture.
 */
export class GpuPanoramaRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly gl: WebGL2RenderingContext | null;
  private readonly program: WebGLProgram | null;
  private readonly volumeTexture: WebGLTexture | null;
  private readonly archTexture: WebGLTexture | null;
  private readonly trackingTexture: WebGLTexture | null;
  private uploadedVolume: Volume | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
    if (!this.gl) { this.program = this.volumeTexture = this.archTexture = this.trackingTexture = null; return; }
    try {
      const gl = this.gl;
      const program = gl.createProgram()!;
      gl.attachShader(program, shader(gl, gl.VERTEX_SHADER, vertex));
      gl.attachShader(program, shader(gl, gl.FRAGMENT_SHADER, fragment));
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? 'shader link failed');
      this.program = program;
      this.volumeTexture = gl.createTexture();
      this.archTexture = gl.createTexture();
      this.trackingTexture = gl.createTexture();
      gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.trackingTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, 1, 1, 0, gl.RED, gl.FLOAT, new Float32Array(1));
      const buffer = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
      const location = gl.getAttribLocation(program, 'position');
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
    } catch {
      this.program = this.volumeTexture = this.archTexture = this.trackingTexture = null;
    }
  }

  /** Forget the uploaded volume so Reset lets the previous case be garbage collected. */
  release() { this.uploadedVolume = null; }

  get available() { return !!this.gl && !!this.program && !!this.volumeTexture && !!this.archTexture; }

  private ensureVolume(volume: Volume): boolean {
    const gl = this.gl!;
    // Most mobile/desktop WebGL2 implementations cap 3D textures. Fall back to
    // the CPU renderer instead of silently truncating a clinical volume.
    if (Math.max(volume.nx, volume.ny, volume.nz) > gl.getParameter(gl.MAX_3D_TEXTURE_SIZE)) return false;
    if (this.uploadedVolume !== volume) {
      const raw = new Uint16Array(volume.data.length);
      for (let i = 0; i < raw.length; i++) raw[i] = Math.max(0, Math.min(65535, Math.round(volume.data[i] + 32768)));
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, this.volumeTexture);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.R16UI, volume.nx, volume.ny, volume.nz, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT, raw);
      this.uploadedVolume = volume;
    }
    return true;
  }

  /** Draw `columns` curve samples (x, y, slabDirX, slabDirY per column) into the currently bound framebuffer. */
  private draw(volume: Volume, curve: Float32Array, columns: number, slab: Slab, window: { center: number; width: number }, tracking?: ArchResult['panorama']) {
    const gl = this.gl!, program = this.program!;
    const samples = Math.max(1, Math.min(25, 2 * Math.floor((slab.thicknessMm / Math.min(volume.sx, volume.sy)) / 2) + 1));
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.archTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, columns, 1, 0, gl.RGBA, gl.FLOAT, curve);
    gl.viewport(0, 0, columns, volume.nz);
    gl.useProgram(program);
    const uniform = (name: string) => gl.getUniformLocation(program, name);
    gl.uniform1i(uniform('volumeTex'), 0); gl.uniform1i(uniform('archTex'), 1);
    gl.uniform1i(uniform('trackingTex'), 3); gl.uniform1i(uniform('trackMandible'), tracking ? 1 : 0);
    if (tracking) {
      gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.trackingTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, columns, volume.nz, 0, gl.RED, gl.FLOAT, tracking.offsets);
    }
    gl.uniform3i(uniform('dimensions'), volume.nx, volume.ny, volume.nz);
    gl.uniform3f(uniform('spacing'), volume.sx, volume.sy, volume.sz);
    gl.uniform1i(uniform('columnCount'), columns);
    gl.uniform1i(uniform('samples'), samples);
    gl.uniform1f(uniform('thickness'), slab.thicknessMm);
    gl.uniform1f(uniform('windowCenter'), window.center);
    gl.uniform1f(uniform('windowWidth'), window.width);
    gl.uniform1i(uniform('projectionMode'), slab.mode === 'max' ? 1 : 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  render(volume: Volume, arch: ArchResult, slab: Slab, window: { center: number; width: number }): boolean {
    if (!this.available) return false;
    const gl = this.gl!;
    try {
      if (!this.ensureVolume(volume)) return false;
      const curve = new Float32Array(arch.points.length * 4);
      for (let i = 0; i < arch.points.length; i++) curve.set([...arch.points[i], ...arch.normals[i]], i * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      if (this.canvas.width !== arch.points.length || this.canvas.height !== volume.nz) { this.canvas.width = arch.points.length; this.canvas.height = volume.nz; }
      this.draw(volume, curve, arch.points.length, slab, window, slab.trackMandible === false ? undefined : arch.panorama);
      return true;
    } catch {
      return false;
    }
  }

  private fbo: WebGLFramebuffer | null = null;
  private fboTexture: WebGLTexture | null = null;
  private fboSize: [number, number] = [0, 0];
  private pixels = new Uint8Array(0);

  /**
   * Straight cross-section on the GPU (same shader as the panorama: a straight "curve" along the arch normal with
   * the slab taken along the arch tangent) read back into a 2D canvas. Replaces a ~75 ms/section CPU render.
   * `frame` is the arch point/normal/tangent at the section; returns false when the CPU path must be used.
   */
  renderCross(target: HTMLCanvasElement, volume: Volume, frame: { p: number[]; n: number[]; t: number[] }, halfWidthMm: number,
    slab: Slab, window: { center: number; width: number }): boolean {
    if (!this.available) return false;
    const gl = this.gl!;
    try {
      if (!this.ensureVolume(volume)) return false;
      const w = Math.max(2, Math.round((2 * halfWidthMm) / Math.min(volume.sx, volume.sy))), h = volume.nz;
      const curve = new Float32Array(w * 4);
      for (let c = 0; c < w; c++) {
        const off = -halfWidthMm + (c / (w - 1)) * 2 * halfWidthMm;
        curve[c * 4] = frame.p[0] + frame.n[0] * off; curve[c * 4 + 1] = frame.p[1] + frame.n[1] * off;
        curve[c * 4 + 2] = frame.t[0]; curve[c * 4 + 3] = frame.t[1];
      }
      if (!this.fbo) { this.fbo = gl.createFramebuffer(); this.fboTexture = gl.createTexture(); }
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
      if (this.fboSize[0] !== w || this.fboSize[1] !== h) {
        gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.fboTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.fboTexture, 0);
        this.fboSize = [w, h];
      }
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) { gl.bindFramebuffer(gl.FRAMEBUFFER, null); return false; }
      this.draw(volume, curve, w, slab, window);
      if (this.pixels.length !== w * h * 4) this.pixels = new Uint8Array(w * h * 4);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, this.pixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      if (target.width !== w || target.height !== h) { target.width = w; target.height = h; }
      const ctx = target.getContext('2d')!;
      const image = ctx.createImageData(w, h), row = w * 4;
      // GL row 0 is the lowest z; canvas row 0 is the top, so flip (high z drawn at the top, like the CPU path).
      for (let y = 0; y < h; y++) image.data.set(this.pixels.subarray((h - 1 - y) * row, (h - y) * row), y * row);
      ctx.putImageData(image, 0, 0);
      return true;
    } catch {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return false;
    }
  }
}
