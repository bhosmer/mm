import * as THREE from 'three'
import * as util from './util.js'

//
//
//

const TEXTURE = new THREE.TextureLoader().load('../examples/textures/sprites/ball.png');

const MATERIAL = new THREE.ShaderMaterial({
  uniforms: {
    color: { value: new THREE.Color(0xffffff) },
    pointTexture: { value: TEXTURE }
  },

  vertexShader: `
  attribute float pointSize;
  attribute vec4 pointColor;
  varying vec4 vColor;

  void main() {
    vColor = pointColor;
    vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
    gl_PointSize = pointSize / -mvPosition.z;
    gl_Position = projectionMatrix * mvPosition;
  }
`,

  fragmentShader: `
  uniform vec3 color;
  uniform sampler2D pointTexture;
  varying vec4 vColor;

  void main() {
    vec4 outColor = texture2D( pointTexture, gl_PointCoord );
    if ( outColor.a < 0.5 ) discard;
    gl_FragColor = outColor * vec4( color * vColor.xyz, 1.0 );
  }`,
})

//
//
//

// https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
// Standard Normal variate using Box-Muller transform.
function gaussianRandom(mean = 0, stdev = 1) {
  let u = 1 - Math.random(); //Converting [0,1) to (0,1)
  let v = Math.random();
  let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  // Transform to the desired mean and standard deviation:
  return z * stdev + mean;
}

// def softmax(x):
//     """Compute softmax values for each sets of scores in x."""
//     e_x = np.exp(x) #
//     return np.exp(x)/sum(np.exp(x))

// note: assumes x is in [0, 1]
function squeeze(base, range, x) {
  return base + range * x
}

function getInitFunc(name, sparsity, base, range) {
  const gate = sparsity > 0 ?
    (f => Math.random() > sparsity ? f() : 0) :
    (f => f())
  switch (name) {
    case 'rows':
      return (i, j, h, w) => gate(() => squeeze(base, range, h > 1 ? i / (h - 1) : 0))
    case 'cols':
      return (i, j, h, w) => gate(() => squeeze(base, range, w > 1 ? j / (w - 1) : 0))
    case 'row major':
      return (i, j, h, w) => gate(() => squeeze(base, range, h * w > 1 ? (i * w + j) / (h * w - 1) : 0))
    case 'col major':
      return (i, j, h, w) => gate(() => squeeze(base, range, h * w > 1 ? (j * h + i) / (h * w) : 0))
    case 'uniform':
      return (i, j, h, w) => gate(() => squeeze(base, range, Math.random()))
    case 'gaussian':
      return (i, j, h, w) => gate(() => squeeze(base, range, gaussianRandom(0.5, 0.5)))
    case 'tril':
      return (i, j, h, w) => gate(() => (j <= i ? 1 : 0))
    case 'triu':
      return (i, j, h, w) => gate(() => (j >= i ? 1 : 0))
    case 'eye':
      return (i, j, h, w) => gate(() => (i == j ? 1 : 0))
    case 'diff':
      return (i, j, h, w) => gate(() => (i == j ? 1 : i == j + 1 ? -1 : 0))
    case 'ones':
      return (i, j, h, w) => gate(() => 1)
    case 'zeros':
      return (i, j, h, w) => gate(() => 0)
    default:
      throw Error(`unrecognized initializer: ${name}`)
  }
}

//
// Array
//
class Array {
  constructor(h, w, f) {
    this.h = h
    this.w = w
    this.data = new Float32Array(h * w)
    this.absmax = 0
    let ptr = 0
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++, ptr++) {
        const x = f(i, j, h, w)
        this.data[ptr] = x
        const absx = Math.abs(x)
        if (absx > this.absmax) {
          this.absmax = absx
        }
      }
    }
  }

  get(i, j) {
    return this.data[this.addr(i, j)]
  }

  set(i, j, x) {
    const oldabsx = Math.abs(this.get(i, j))
    this.data[this.addr(i, j)] = x
    const absx = Math.abs(x)
    if (absx > this.absmax) {
      this.absmax = absx
    } else if (absx < this.absmax && oldabsx == this.absmax) {
      this.absmax = this.data.reduce(function (acc, x) {
        const absx = Math.abs(x)
        return absx > acc ? absx : acc
      })
    }
  }

  addr(i, j) {
    return i * this.w + j
  }
}

//
// Mat
//
class Mat {
  ELEM_SIZE = 2048
  ELEM_SAT = 1.0
  ELEM_LIGHT = 0.6

  sizeFromData(x) {
    if (isNaN(x)) {
      return 0
    }
    const zsize = this.zero_size * this.ELEM_SIZE
    const range = (1 - this.zero_size) * this.ELEM_SIZE
    const size = zsize + range * Math.abs(x) / Math.max(this.container.global_absmax, 0.01)
    return size
  }

  setElemHSL(a, i, x, s = this.ELEM_SAT, l = this.ELEM_LIGHT) {
    if (isNaN(x)) {
      const c = new THREE.Color().setHSL(0, 0, 0)
      c.toArray(a, i * 3)
      return
    }

    const gap = (x == 0 ? 1 : Math.sign(x)) * this.hue_gap
    const h = (this.zero_hue + gap + (x / this.container.global_absmax * this.hue_spread)) % 1

    const lrange = this.max_light - this.zero_light
    l = this.zero_light + Math.abs(x) * lrange / Math.max(this.data.absmax, 0.01)

    const c = new THREE.Color().setHSL(h, s, l)
    c.toArray(a, i * 3)
  }

  static fromInit(h, w, init, container) {
    return new Mat(h, w, new Array(h, w, init), container)
  }

  constructor(h, w, data, container) {
    this.container = container
    this.zero_hue = container.params['zero hue']
    this.zero_size = container.params['zero size']
    this.zero_light = container.params['zero light']
    this.max_light = container.params['max light']
    this.hue_gap = container.params['hue gap']
    this.hue_spread = container.params['hue spread']
    this.h = h
    this.w = w
    this.data = data
    let sizes = new Float32Array(this.numel())
    let colors = new Float32Array(this.numel() * 3)
    let points = []
    for (let i = 0; i < this.h; i++) {
      for (let j = 0; j < this.w; j++) {
        points.push(new THREE.Vector3(j, i, 0))
        sizes[this.data.addr(i, j)] = this.sizeFromData(this.getData(i, j))
        this.setElemHSL(colors, this.data.addr(i, j), this.getData(i, j))
      }
    }
    const g = new THREE.BufferGeometry().setFromPoints(points);
    g.setAttribute('pointSize', new THREE.Float32BufferAttribute(sizes, 1))
    g.setAttribute('pointColor', new THREE.Float32BufferAttribute(colors, 3))
    this.points = new THREE.Points(g, MATERIAL)
    this.group = new THREE.Group()
    this.group.add(this.points)
  }

  numel() {
    return this.h * this.w
  }

  hideAll() {
    for (let i = 0; i < this.h; i++) {
      for (let j = 0; j < this.w; j++) {
        this.setSize(i, j, this.sizeFromData(NaN))
        this.setHSL(i, j, NaN)
      }
    }
  }

  showAll() {
    for (let i = 0; i < this.h; i++) {
      for (let j = 0; j < this.w; j++) {
        this.show(i, j)
      }
    }
  }

  show(i, j) {
    this.setSize(i, j, this.sizeFromData(this.getData(i, j)))
    this.setHSL(i, j, this.getData(i, j))
  }

  setSize(i, j, x) {
    this.points.geometry.attributes.pointSize.array[this.data.addr(i, j)] = x
    this.points.geometry.attributes.pointSize.needsUpdate = true
  }

  setHSL(i, j, h, s = this.ELEM_SAT, l = this.ELEM_LIGHT) {
    this.setElemHSL(this.points.geometry.attributes.pointColor.array, this.data.addr(i, j), h, s, l)
    this.points.geometry.attributes.pointColor.needsUpdate = true
  }

  getColor(i, j) {
    const c = new THREE.Color()
    return c.fromArray(this.points.geometry.attributes.pointColor.array, this.data.addr(i, j) * 3)
  }

  setColor(i, j, c) {
    c.toArray(this.points.geometry.attributes.pointColor.array, this.data.addr(i, j) * 3)
    this.points.geometry.attributes.pointColor.needsUpdate = true
  }

  getData(i, j) {
    return this.data.get(i, j)
  }

  setData(i, j, x) {
    this.data.set(i, j, x)
    this.setSize(i, j, this.sizeFromData(x))
    this.setHSL(i, j, x)
    this.container.setAbsmax()
  }

  bumpColor(i, j, up) {
    if (up) {
      let c = this.getColor(i, j)
      const bump = new THREE.Color(0x808080)
      c.add(bump)
      this.setColor(i, j, c)
    } else {
      this.setHSL(i, j, this.getData(i, j))
    }
  }

  bumpRowColor(i, up) {
    for (let j = 0; j < this.w; j++) {
      this.bumpColor(i, j, up)
    }
  }

  bumpColumnColor(j, up) {
    for (let i = 0; i < this.h; i++) {
      this.bumpColor(i, j, up)
    }
  }

  setGuide(enabled) {
    if (enabled) {
      if (!this.guide) {
        const guide = util.rowguide(this.h, 0.5, this.w)
        this.group.add(guide)
        this.guide = guide
      }
    } else {
      if (this.guide) {
        this.group.remove(this.guide)
        this.guide = undefined
      }
    }
  }

  setNameLegend(enabled, name, getText, color, size) {
    if (enabled) {
      if (!this.name_legend) {
        const legend = getText(name, color, size)
        const { h: hleg, w: wleg } = util.bbhw(legend.geometry)
        legend.geometry.rotateY(Math.PI)
        legend.geometry.rotateZ(Math.PI)
        legend.geometry.translate(util.center(this.w - 1, wleg), hleg + util.center(this.h - 1, hleg), -(1 + hleg / 2))
        this.group.add(legend)
        this.name_legend = legend
      }
    } else {
      if (this.name_legend) {
        this.group.remove(this.name_legend)
        this.name_legend = undefined
      }
    }
  }

  setHeightLegend(enabled, name, left, getText, color, size) {
    if (enabled) {
      if (!this.height_legend) {
        const legend = getText(`${name} = ${this.h}`, color, size)
        const { h: hleg, w: wleg } = util.bbhw(legend.geometry)
        legend.geometry.rotateX(Math.PI)
        const zrot = (left ? -1 : 1) * Math.PI / 2
        legend.geometry.rotateZ(zrot)
        const spacer = 0.25
        const xoff = left ? -hleg * 1.5 - spacer : this.w - 1 + hleg + spacer
        const yoff = left ? wleg + util.center(this.h - 1, wleg) : util.center(this.h - 1, wleg)
        legend.geometry.translate(xoff, yoff, 0)
        this.group.add(legend)
        this.height_legend = legend
      }
    } else {
      if (this.height_legend) {
        this.group.remove(this.height_legend)
        this.height_legend = undefined
      }
    }
  }

  setWidthLegend(enabled, name, top, getText, color, size) {
    if (enabled) {
      if (!this.width_legend) {
        const legend = getText(`${name} = ${this.w}`, color, size)
        const { h: hleg, w: wleg } = util.bbhw(legend.geometry)
        legend.geometry.rotateX(Math.PI)
        const spacer = 0.25
        const xoff = util.center(this.w - 1, wleg)
        const yoff = top ? -hleg * 2 - spacer : this.h - 1 + hleg * 1.5 + spacer
        legend.geometry.translate(xoff, yoff, 0)
        this.group.add(legend)
        this.width_legend = legend
      }
    } else {
      if (this.width_legend) {
        this.group.remove(this.width_legend)
        this.width_legend = undefined
      }
    }
  }

}

//
// MatMul
//
export class MatMul {
  constructor(params, getText) {
    this.getText = getText
    this.params = { ...params }
    this.group = new THREE.Group()

    this.H = params.I
    this.D = params.J
    this.W = params.K

    this.init_base = params['init min']
    this.init_range = Math.max(0, params['init max'] - params['init min'])

    this.initLeftData();
    this.initRightData();
    this.initResultData();

    this.initVis(params)
  }

  initLeftData() {
    if (this.params.left) {
      this.left_data = this.params.left.data
      return
    }
    const init = this.params['left init']
    const sparsity = this.params['left sparsity']
    const left_init = getInitFunc(init, sparsity, this.init_base, this.init_range);
    this.left_data = new Array(this.H, this.D, left_init);
  }

  initRightData() {
    if (this.params.right) {
      this.right_data = this.params.right.data
      return
    }
    const init = this.params['right init']
    const sparsity = this.params['right sparsity']
    const right_init = getInitFunc(init, sparsity, this.init_base, this.init_range);
    this.right_data = new Array(this.D, this.W, right_init);
  }

  initResultData() {
    const result_init = (y, x, h, w) => this._result_val(this.left_data, this.right_data, y, x);
    this.result_data = new Array(this.H, this.W, result_init);
  }

  initVis(params) {
    if (params) {
      this.params = { ...params }
    }
    this.group.clear()

    this._setAbsmax(this.left_data, this.right_data, this.result_data)

    this.initLeftVis();
    this.initRightVis();
    this.initResultVis();

    this.animation = 'none'
    this.setAnimation(this.params.animation)

    this.setPosition();
  }

  setPosition() {
    // center cube on 0,0,0 if no pos given
    // note: don't save into params
    const pos = this.params.pos ? this.params.pos :
      new THREE.Vector3(-(this.W - 1) / 2, (this.H - 1) / 2, -(this.D - 1) / 2)
    this.group.position.x = pos.x
    this.group.position.y = pos.y
    this.group.position.z = pos.z
  }

  initLeftVis() {
    if (this.params.left) {
      this.left = this.params.left
      return
    }
    if (this.left) {
      this.group.remove(this.left.group)
    }
    this.left = new Mat(this.H, this.D, this.left_data, this);
    this.left.group.rotation.y = Math.PI / 2;
    this.left.group.rotation.z = Math.PI;
    if (this.params.left_rot) {
      Object.keys(this.params.left_rot).map(k => this.left.group.rotation[k] += this.params.left_rot[k])
    }
    this.left.group.position.x = -1;
    if (this.params.left_pos) {
      Object.keys(this.params.left_pos).map(k => this.left.group.position[k] += this.params.left_pos[k])
    }
    this.left.setGuide(this.params.guides)
    this.setLeftLegends(this.params.legends)
    this.group.add(this.left.group)
  }

  initRightVis() {
    if (this.params.right) {
      this.right = this.params.right
      return
    }
    if (this.right) {
      this.group.remove(this.right.group)
    }
    this.right = new Mat(this.D, this.W, this.right_data, this);
    this.right.group.rotation.x = Math.PI / 2;
    if (this.params.right_rot) {
      Object.keys(this.params.right_rot).map(k => this.right.group.rotation[k] += this.params.right_rot[k])
    }
    this.right.group.position.y = 1;
    if (this.params.right_pos) {
      Object.keys(this.params.right_pos).map(k => this.right.group.position[k] += this.params.right_pos[k])
    }
    this.right.setGuide(this.params.guides)
    this.setRightLegends(this.params.legends)
    this.group.add(this.right.group)
  }

  initResultVis() {
    if (this.result) {
      this.group.remove(this.result.group)
    }
    this.result = new Mat(this.H, this.W, this.result_data, this);
    this.result.group.rotation.x = Math.PI;
    if (this.params.result_rot) {
      Object.keys(this.params.result_rot).map(k => this.result.group.rotation[k] += this.params.result_rot[k])
    }
    this.result.group.position.z = this.D;
    if (this.params.result_pos) {
      Object.keys(this.params.result_pos).map(k => this.result.group.position[k] += this.params.result_pos[k])
    }
    this.result.setGuide(this.params.guides)
    this.setResultLegends(this.params.legends)
    this.group.add(this.result.group);
  }

  setEpilog(epilog) {
    this.params.epilog = epilog
    this.initResultData()
    this.initVis()
  }

  setI(i) {
    this.H = this.params.I = i
    this.initLeft()
  }

  initLeft(params = undefined) {
    if (params) {
      this.params = { ...params }
    }
    this.initLeftData();
    this.initResultData();
    this._setAbsmax(this.left_data, this.right_data, this.result_data);
    this.initVis(this.params);
  }

  setK(k) {
    this.W = this.params.K = k
    this.initRight();
  }

  initRight(params = undefined) {
    if (params) {
      this.params = { ...params }
    }
    this.initRightData();
    this.initResultData();
    this._setAbsmax(this.left_data, this.right_data, this.result_data);
    this.initVis(this.params);
  }

  setAnimation(animation) {
    this.animation = animation
    if (this.animation == 'dotprod') {
      this.result.hideAll()
      const dotprod_init = (y, x, h, w) => this.dotprod_val(0, 0, x)
      this.dotprod = Mat.fromInit(1, this.D, dotprod_init, this)
      this.dotprod.points.rotation.y = -Math.PI / 2
      this.group.add(this.dotprod.points)
      this.bump = this.bump_dotprod
      this.curi = this.H - 1
      this.curk = this.W - 1
    } else if (this.animation == 'itemwise') {
      this.result.hideAll()
      const itemwise_init = (y, x, h, w) => this.dotprod_val(0, 0, x)
      this.itemwise = Mat.fromInit(1, 1, itemwise_init, this)
      this.group.add(this.itemwise.points)
      this.bump = this.bump_itemwise
      this.curi = this.H - 1
      this.curj = this.D - 1
      this.curk = this.W - 1
    } else if (this.animation == 'mvprod') {
      this.result.hideAll()
      const mvprod_init = (y, x, h, w) => this.dotprod_val(0, y, x)
      this.mvprod = Mat.fromInit(this.H, this.D, mvprod_init, this)
      this.mvprod.points.rotation.y = Math.PI / 2
      this.mvprod.points.rotation.z = Math.PI
      this.group.add(this.mvprod.points)
      this.bump = this.bump_mvprod
      this.curk = this.W - 1
    } else if (this.animation == 'vmprod') {
      this.result.hideAll()
      const vmprod_init = (y, x, h, w) => this.dotprod_val(y, 0, x)
      this.vmprod = Mat.fromInit(this.D, this.W, vmprod_init, this)
      this.vmprod.points.rotation.x = Math.PI / 2
      this.group.add(this.vmprod.points)
      this.bump = this.bump_vmprod
      this.curi = this.H - 1
    } else if (this.animation == 'none') {
      this.result.showAll()
    }
  }

  dotprod_val(i, j, k) {
    return this.left.getData(i, k) * this.right.getData(k, j)
  }

  _result_val(a, b, i, j) {
    let x = 0.0
    const n = this.animation == 'itemwise' ? this.curk : a.w
    for (let k = 0; k < n; k++) {
      x += a.get(i, k) * b.get(k, j)
    }
    const epi = this.params.epilog
    return epi == 'x/J' ? x / this.D :
      epi == 'x/sqrt(J)' ? x / Math.sqrt(this.D) :
        epi == 'tanh(x)' ? Math.tanh(x) :
          epi == 'relu(x)' ? Math.max(0, x) :
            x
  }

  result_val(i, j) {
    return this._result_val(this.left.data, this.right.data, i, j)
  }

  _setAbsmax(a, b, c) {
    this.global_absmax = Math.max(a.absmax, b.absmax, c.absmax)
  }

  setAbsmax() {
    this._setAbsmax(this.left.data, this.right.data, this.result.data)
  }

  bump_vmprod() {
    const oldi = this.curi

    if (oldi < this.H - 1) {
      this.curi += 1
    } else {
      this.curi = 0
    }

    const i = this.curi

    // update result face
    if (this.curi == 0) {
      this.result.hideAll()
    }
    for (let k = 0; k < this.W; k++) {
      this.result.show(i, k)
    }

    this.left.bumpRowColor(oldi, false)
    this.left.bumpRowColor(i, true)

    // move and recolor dot product vector
    this.vmprod.points.position.y = -this.left.points.geometry.attributes.position.array[i * this.D * 3 + 1]
    for (let x = 0; x < this.W; x++) {
      for (let z = 0; z < this.D; z++) {
        this.vmprod.setData(z, x, this.dotprod_val(i, x, z))
      }
    }
  }

  bump_mvprod() {
    const oldk = this.curk

    if (oldk < this.W - 1) {
      this.curk += 1
    } else {
      this.curk = 0
    }

    const k = this.curk

    // update result face
    if (this.curk == 0) {
      this.result.hideAll()
    }
    for (let i = 0; i < this.H; i++) {
      this.result.show(i, k)
    }

    this.right.bumpColumnColor(oldk, false)
    this.right.bumpColumnColor(k, true)

    // move and recolor dot product vector
    this.mvprod.points.position.x = this.right.points.geometry.attributes.position.array[k * 3]
    for (let y = 0; y < this.H; y++) {
      for (let z = 0; z < this.D; z++) {
        this.mvprod.setData(y, z, this.dotprod_val(y, k, z))
      }
    }
  }

  bump_dotprod() {
    const oldi = this.curi
    const oldk = this.curk

    if (oldk < this.W - 1) {
      this.curk += 1
    } else {
      this.curk = 0
      this.curi = oldi < this.H - 1 ? this.curi + 1 : 0
    }

    const i = this.curi
    const k = this.curk

    // update result face
    if (i == 0 && k == 0) {
      this.result.hideAll()
    }
    this.result.show(i, k)

    // hilight operand row/cols
    if (oldk != k) {
      if (oldk >= 0) {
        this.right.bumpColumnColor(oldk, false)
      }
      this.right.bumpColumnColor(k, true)
    }

    if (oldi != i) {
      if (oldi >= 0) {
        this.left.bumpRowColor(oldi, false)
      }
      this.left.bumpRowColor(i, true)
    }

    // move and recolor dot product vector
    this.dotprod.points.position.x = this.right.points.geometry.attributes.position.array[k * 3]
    this.dotprod.points.position.y = -this.left.points.geometry.attributes.position.array[i * this.D * 3 + 1]
    for (let z = 0; z < this.dotprod.numel(); z++) {
      this.dotprod.setData(0, z, this.dotprod_val(i, k, z))
    }
  }

  bump_itemwise() {
    const oldi = this.curi
    const oldj = this.curj
    const oldk = this.curk

    if (oldj < this.D - 1) {
      this.curj += 1
    } else {
      this.curj = 0
      if (oldk < this.W - 1) {
        this.curk += 1
      } else {
        this.curk = 0
        this.curi = oldi < this.H - 1 ? this.curi + 1 : 0
      }
    }

    const i = this.curi
    const j = this.curj
    const k = this.curk

    // update result face
    if (i == 0 && k == 0) {
      this.result.hideAll()
    }
    this.result.show(i, k)

    // hilight operand row/cols
    this.left.bumpColor(oldi, oldj, false)
    this.left.bumpColor(i, j, true)

    this.right.bumpColor(oldj, oldk, false)
    this.right.bumpColor(j, k, true)

    // move and recolor multiple item
    this.itemwise.points.position.x = this.right.points.geometry.attributes.position.array[k * 3]
    this.itemwise.points.position.y = -this.left.points.geometry.attributes.position.array[i * this.D * 3 + 1]
    this.itemwise.points.position.z = this.left.points.geometry.attributes.position.array[j * 3]
    this.itemwise.setData(0, 0, this.dotprod_val(i, k, j))
  }

  // TODO devolve to Mat
  setGuides(enabled) {
    this.params.guides = enabled

    if (!this.params.left) {
      this.left.setGuide(enabled)
    }
    if (!this.params.right) {
      this.right.setGuide(enabled)
    }
    this.result.setGuide(enabled)
  }

  getLeftName() {
    return this.params.left_name ? this.params.left_name : "X"
  }

  getRightName() {
    return this.params.right_name ? this.params.right_name : "Y"
  }

  getResultName() {
    return this.params.result_name ? this.params.result_name : "XY"
  }

  getHeightName() {
    return this.params.height_name ? this.params.height_name : "i"
  }

  getDepthName() {
    return this.params.depth_name ? this.params.depth_name : "j"
  }

  getWidthName() {
    return this.params.width_name ? this.params.width_name : "k"
  }

  getNameColor() {
    return this.params.name_color ? this.params.name_color : 0xccccff
  }

  getDimColor() {
    return this.params.dim_color ? this.params.dim_color : 0x00aaff
  }

  getNameSize() {
    return (this.H + this.D + this.W) / 32
  }

  getDimSize() {
    return (this.H + this.D + this.W) / 75
  }

  // TODO this is clumsy, devolve to Mat

  setLeftLegends(enabled) {
    this.left.setNameLegend(enabled, this.getLeftName(), this.getText, this.getNameColor(), this.getNameSize())
    this.left.setHeightLegend(enabled, this.getHeightName(), true, this.getText, this.getDimColor(), this.getDimSize())
    this.left.setWidthLegend(enabled, this.getDepthName(), false, this.getText, this.getDimColor(), this.getDimSize())
  }

  setRightLegends(enabled) {
    this.right.setNameLegend(enabled, this.getRightName(), this.getText, this.getNameColor(), this.getNameSize())
    this.right.setHeightLegend(enabled, this.getDepthName(), false, this.getText, this.getDimColor(), this.getDimSize())
    this.right.setWidthLegend(enabled, this.getWidthName(), true, this.getText, this.getDimColor(), this.getDimSize())
  }

  setResultLegends(enabled) {
    this.result.setNameLegend(enabled, this.getResultName(), this.getText, this.getNameColor(), this.getNameSize())
    this.result.setHeightLegend(enabled, this.getHeightName(), false, this.getText, this.getDimColor(), this.getDimSize())
    this.result.setWidthLegend(enabled, this.getWidthName(), false, this.getText, this.getDimColor(), this.getDimSize())
  }


  setLegends(enabled) {
    this.params.legends = enabled
    this.setLeftLegends(enabled)
    this.setRightLegends(enabled)
    this.setResultLegends(enabled)
  }

}

//
// Attn (Q @ K) @ V
//

export class Attn {
  constructor(params, getText) {
    this.getText = getText
    this.params = { ...params }
    this.group = new THREE.Group()

    // TODO passed in
    const mm1_params = { ...params }
    const mm2_params = { ...params }

    if (mm1_params.I != mm2_params.I) {
      throw Error(`Attn: mm1_params.I ${mm1_params.I} mm2_params.I ${mm2_params.I}`)
    }
    if (mm1_params.K != mm2_params.J) {
      throw Error(`Attn: mm1_params.K ${mm1_params.K} mm2_params.J ${mm2_params.J}`)
    }

    this.H = mm1_params.I
    this.D = mm1_params.J + mm2_params.K
    this.W = mm1_params.K

    this.mm1_params = mm1_params
    this.initmm1()
    this.mm2_params = mm2_params
    this.initmm2()

    this.setPosition()
  }

  setPosition() {
    // center cube on 0,0,0 if no pos given
    // note: don't save into params
    const pos = this.params.pos ? this.params.pos :
      new THREE.Vector3(-this.W / 2, this.H / 2, -this.D / 2)
    this.group.position.x = pos.x
    this.group.position.y = pos.y
    this.group.position.z = pos.z
  }

  initmm1() {
    if (this.mm1) {
      this.group.remove(this.mm1.group)
    }
    this.mm1_params.left_name = "Q"
    this.mm1_params.right_name = "K^T"
    this.mm1_params.result_name = "attn"
    this.mm1_params.pos = new THREE.Vector3(0, 0, 0)
    this.mm1 = new MatMul(this.mm1_params, this.getText, this.group)
    this.group.add(this.mm1.group)
  }

  initmm2() {
    if (this.mm2) {
      this.group.remove(this.mm2.group)
    }
    this.mm2_params.left = this.mm1.result
    this.mm2_params.right_name = "V"
    this.mm2_params.result_name = "out"
    this.mm2_params.pos = new THREE.Vector3(0, 0, this.mm1.D + 1)
    this.mm2_params.right_rot = new THREE.Vector3(Math.PI, 0, -Math.PI / 2)
    this.mm2_params.result_pos = new THREE.Vector3(this.W, 0, -this.mm1_params.K)
    this.mm2_params.result_rot = new THREE.Vector3(0, Math.PI / 2, 0)
    this.mm2 = new MatMul(this.mm2_params, this.getText, this.group)
    this.group.add(this.mm2.group)
  }

  bump() {
    this.mm1.bump()
    this.mm2.bump()
  }

  setI(i) {
    this.H = this.params.I = i
    this.mm1_params.I = i
    this.mm1.setI(i)
    this.mm2_params.I = i
    this.mm2.setI(i)
    this.setPosition()
  }

  setK(k) {
    this.W = this.params.K = k
    this.mm1_params.K = k
    this.mm1.setK(k)
    this.mm2_params.J = k
    this.initmm2()
    this.setPosition()
  }

  setGuides(enabled) {
    this.mm1.setGuides(enabled)
    this.mm2.setGuides(enabled)
  }

  setLegends(enabled) {
    this.mm1.setLegends(enabled)
    this.mm2.setLegends(enabled)
  }
}

//
// MLP W1 @ (W0 @ X)
//

export class MLP {
  constructor(params, getText) {
    this.getText = getText
    this.group = new THREE.Group()

    // TODO passed in
    const mm1_params = { ...params }
    const mm2_params = { ...params }

    if (mm1_params.I != mm2_params.K) {
      throw Error(`MLP: mm1_params.I ${mm1_params.I} mm2_params.K ${mm2_params.K}`)
    }
    if (mm1_params.K != mm2_params.J) {
      throw Error(`MLP: mm1_params.K ${mm1_params.K} mm2_params.J ${mm2_params.J}`)
    }

    this.H = mm1_params.I
    this.D = mm1_params.J + mm2_params.I
    this.W = mm1_params.K

    // TODO offset from parent pos
    this.mm1_params = mm1_params
    this.mm1_params.left_name = "W0"
    this.mm1_params.right_name = "X"
    this.mm1_params.result_name = "W0 @ X"
    this.mm1_params.pos = new THREE.Vector3(-this.W / 2, this.H / 2, -this.D / 2)
    this.mm1 = new MatMul(this.mm1_params, getText, this.group)
    this.group.add(this.mm1.group)

    this.mm2_params = mm2_params
    this.mm2_params.left_name = "W1"
    this.mm2_params.result_name = "out"
    this.mm2_params.pos = new THREE.Vector3(-this.W / 2, this.H / 2, -this.D / 2 + mm1_params.J)
    this.mm2_params.left_pos = new THREE.Vector3(0, 0, 1)
    this.mm2_params.left_rot = new THREE.Vector3(Math.PI, Math.PI, -Math.PI / 2)
    this.mm2_params.right = this.mm1.result
    this.mm2_params.result_pos = new THREE.Vector3(0, -this.H, -mm2_params.I + 1) // bot
    this.mm2_params.result_rot = new THREE.Vector3(-Math.PI / 2, 0, 0)
    this.mm2 = new MatMul(this.mm2_params, getText, this.group)
    this.group.add(this.mm2.group)

  }

  bump() {
    this.mm1.bump()
    this.mm2.bump()
  }

  setGuides(enabled) {
    this.mm1.setGuides(enabled)
    this.mm2.setGuides(enabled)
  }

  setLegends(enabled) {
    this.mm1.setLegends(enabled)
    this.mm2.setLegends(enabled)
  }
}
