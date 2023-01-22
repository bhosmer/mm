import * as THREE from 'three'
import * as util from './util.js'

//
//
//

const TEXTURE = new THREE.TextureLoader().load('../examples/textures/sprites/ball.png')

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
  let u = 1 - Math.random() //Converting [0,1) to (0,1)
  let v = Math.random()
  let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
  // Transform to the desired mean and standard deviation:
  return z * stdev + mean
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
// rules:
//
// mats always have (0, 0, 0) as their minimum point
// cubes also
// cubes are either concave or convex
// convex: 
//
//

// move
function orientFromParams(params) {
  if (params.orient) {
    return params.orient
  }
  const x = params['left/right'] == 'left' ? -1 : 1
  const y = params['up/down'] == 'down' ? -1 : 1
  const z = params['front/back'] == 'back' ? -1 : 1
  return { x: x, y: y, z: z }
}

function rotFromOrient(orient) {
  return {
    x: orient.x == -1 ? Math.PI : 0,
    y: orient.y == -1 ? Math.PI : 0,
    z: orient.z == -1 ? Math.PI : 0,
  }
}

//
// Mat
//
export class Mat {
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

  // --------

  // TODO 
  // param plumbing

  static fromParams(h, w, params, getText) {
    const data = Mat.dataFromParams(h, w, params)

    const m = new Mat(h, w, data, undefined, params)

    m.setGuides(params.guides)

    // this.setLeftLegends(params.legends)
    const custom = params.legend ? params.legend : {}
    const defaults = { name: "X", height: "i", width: "j", hleft: true, wtop: false }
    const props = { ...m.getLegendProps(h, w, params), ...defaults, ...custom }
    m.setLegends(params.legends, props, getText)

    return m
  }

  static dataFromParams(h, w, params) {
    const init_base = params['init min']
    const init_range = Math.max(0, params['init max'] - params['init min'])
    const init_name = params['left init']
    const sparsity = params['left sparsity']
    const init = getInitFunc(init_name, sparsity, init_base, init_range)
    return new Array(h, w, init)
  }

  getLegendProps(h, w, params) {
    const custom = params.legend_props ? params.legend_props : {}
    const gm = Math.sqrt(h * w)
    const defaults = {
      name_color: 0xccccff,
      name_size: gm / 16,
      dim_color: 0x00aaff,
      dim_size: gm / 32,
    }
    return { ...defaults, ...custom }
  }

  // --------

  constructor(h, w, data, container, params) {
    if (container) {
      this.container = container
      if (params) {
        throw Error('passed both container and params to Mat')
      }
      this.orient = orientFromParams(this.container.params)
    } else {
      this.container = this
      this.global_absmax = data.absmax
      this.setAbsmax = () => this.global_absmax = data.absmax
      if (!params) {
        throw Error('passed neither container nor params to Mat')
      }
      this.params = { ...params }
      this.orient = orientFromParams(this.params)
    }

    this.zero_hue = this.container.params['zero hue']
    this.zero_size = this.container.params['zero size']
    this.zero_light = this.container.params['zero light']
    this.max_light = this.container.params['max light']
    this.hue_gap = this.container.params['hue gap']
    this.hue_spread = this.container.params['hue spread']
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
    const g = new THREE.BufferGeometry().setFromPoints(points)
    g.setAttribute('pointSize', new THREE.Float32BufferAttribute(sizes, 1))
    g.setAttribute('pointColor', new THREE.Float32BufferAttribute(colors, 3))
    this.points = new THREE.Points(g, MATERIAL)
    this.group = new THREE.Group()
    this.group.add(this.points)

    this.group.rotation.setFromVector3(rotFromOrient(this.orient))
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

  setGuides(enabled) {
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

  setLegends(enabled, props, getText) {
    if (enabled) {
      if (!this.legends) {
        this.legends = new THREE.Group()
        if (props.name) {
          const name = getText(props.name, props.name_color, props.name_size)
          const { h, w } = util.bbhw(name.geometry)
          name.geometry.rotateZ(Math.PI)
          name.geometry.rotateY(Math.PI)
          name.geometry.translate(util.center(this.w - 1, w), h + util.center(this.h - 1, h), -(1 + h / 2))
          this.legends.add(name)
        }
        if (props.height) {
          const height = getText(`${props.height} = ${this.h}`, props.dim_color, props.dim_size)
          const { h, w } = util.bbhw(height.geometry)
          height.geometry.rotateX(Math.PI)
          const zrot = (props.hleft ? -1 : 1) * Math.PI / 2
          height.geometry.rotateZ(zrot)
          const spacer = 0.5
          const xoff = props.hleft ? -h * 1 - spacer : this.w - 1 + h + spacer
          const yoff = props.hleft ? w + util.center(this.h - 1, w) : util.center(this.h - 1, w)
          height.geometry.translate(xoff, yoff, 0)
          this.legends.add(height)
        }
        if (props.width) {
          const width = getText(`${props.width} = ${this.w}`, props.dim_color, props.dim_size)
          const { h, w } = util.bbhw(width.geometry)
          width.geometry.rotateX(Math.PI)
          const spacer = 0.5
          const xoff = util.center(this.w - 1, w)
          const yoff = props.wtop ? -h * 1 - spacer : this.h - 1 + h * 1.5 + spacer
          width.geometry.translate(xoff, yoff, 0)
          this.legends.add(width)
        }
        this.group.add(this.legends)
      }
    } else {
      if (this.legends) {
        this.group.remove(this.legends)
        this.legends = undefined
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

    this.initData()
    this.initVis()
  }

  initData() {
    this.initLeftData()
    this.initRightData()
    this.initResultData()
  }

  initLeftData() {
    if (this.params.left) {
      this.left_data = this.params.left.data
      return
    }
    const init = this.params['left init']
    const sparsity = this.params['left sparsity']
    const left_init = getInitFunc(init, sparsity, this.init_base, this.init_range)
    this.left_data = new Array(this.H, this.D, left_init)
  }

  initRightData() {
    if (this.params.right) {
      this.right_data = this.params.right.data
      return
    }
    const init = this.params['right init']
    const sparsity = this.params['right sparsity']
    const right_init = getInitFunc(init, sparsity, this.init_base, this.init_range)
    this.right_data = new Array(this.D, this.W, right_init)
  }

  initResultData() {
    const result_init = (y, x, h, w) => this._result_val(this.left_data, this.right_data, y, x)
    this.result_data = new Array(this.H, this.W, result_init)
  }

  initVis(params = undefined) {
    if (params) {
      this.params = { ...params }
    }
    this.group.clear()

    this._setAbsmax(this.left_data, this.right_data, this.result_data)

    this.initLeftVis()
    this.initRightVis()
    this.initResultVis()

    this.animation = 'none'
    this.setAnimation(this.params.animation)

    this.setPosition()
  }

  setPosition() {
    // center cube on 0,0,0 if no pos given
    // note: don't save into params
    const pos = this.params.pos ? this.params.pos :
      new THREE.Vector3(-(this.W - 1) / 2, (this.H - 1) / 2, -(this.D - 1) / 2)
    this.group.position.x = pos.x
    this.group.position.y = pos.y
    this.group.position.z = pos.z
    const rot = this.params.rot ? this.params.rot : new THREE.Vector3(0, 0, 0)
    this.group.rotation.x = rot.x
    this.group.rotation.y = rot.y
    this.group.rotation.z = rot.z
  }

  initLeftVis() {
    if (this.params.left) {
      this.left = this.params.left
      return
    }
    if (this.left) {
      this.group.remove(this.left.group)
    }
    this.left = new Mat(this.H, this.D, this.left_data, this)
    this.left.group.rotation.y = Math.PI / 2
    this.left.group.rotation.z = Math.PI
    if (this.params.left_rot) {
      Object.keys(this.params.left_rot).map(k => this.left.group.rotation[k] += this.params.left_rot[k])
    }
    this.left.group.position.x = -1
    if (this.params.left_pos) {
      Object.keys(this.params.left_pos).map(k => this.left.group.position[k] += this.params.left_pos[k])
    }
    this.left.setGuides(this.params.guides)
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
    this.right = new Mat(this.D, this.W, this.right_data, this)
    this.right.group.rotation.x = Math.PI / 2
    if (this.params.right_rot) {
      Object.keys(this.params.right_rot).map(k => this.right.group.rotation[k] += this.params.right_rot[k])
    }
    this.right.group.position.y = 1
    if (this.params.right_pos) {
      Object.keys(this.params.right_pos).map(k => this.right.group.position[k] += this.params.right_pos[k])
    }
    this.right.setGuides(this.params.guides)
    this.setRightLegends(this.params.legends)
    this.group.add(this.right.group)
  }

  initResultVis() {
    if (this.result) {
      this.group.remove(this.result.group)
    }
    this.result = new Mat(this.H, this.W, this.result_data, this)
    this.result.group.rotation.x = Math.PI
    if (this.params.result_rot) {
      Object.keys(this.params.result_rot).map(k => this.result.group.rotation[k] += this.params.result_rot[k])
    }
    this.result.group.position.z = this.D
    if (this.params.result_pos) {
      Object.keys(this.params.result_pos).map(k => this.result.group.position[k] += this.params.result_pos[k])
    }
    this.result.setGuides(this.params.guides)
    this.setResultLegends(this.params.legends)
    this.group.add(this.result.group)
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
    this.initLeftData()
    this.initResultData()
    this._setAbsmax(this.left_data, this.right_data, this.result_data)
    this.initVis()
  }

  setK(k) {
    this.W = this.params.K = k
    this.initRight()
  }

  initRight(params = undefined) {
    if (params) {
      this.params = { ...params }
    }
    this.initRightData()
    this.initResultData()
    this._setAbsmax(this.left_data, this.right_data, this.result_data)
    this.initVis()
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
    const n = a.w
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

  setGuides(enabled) {
    this.params.guides = enabled
    if (!this.params.left) {
      this.left.setGuides(enabled)
    }
    if (!this.params.right) {
      this.right.setGuides(enabled)
    }
    this.result.setGuides(enabled)
  }

  getLegendProps() {
    const custom = this.params.legend_props ? this.params.legend_props : {}
    const sa_geo = Math.sqrt(this.H * this.D) + Math.sqrt(this.D * this.W) + Math.sqrt(this.H * this.W)
    const defaults = {
      name_color: 0xccccff,
      name_size: sa_geo / 32,
      dim_color: 0x00aaff,
      dim_size: sa_geo / 96,
    }
    return { ...defaults, ...custom }
  }

  setLeftLegends(enabled) {
    const custom = this.params.left_legend ? this.params.left_legend : {}
    const defaults = { name: "X", height: "i", width: "j", hleft: true, wtop: false }
    const props = { ...this.getLegendProps(), ...defaults, ...custom }
    this.left.setLegends(enabled, props, this.getText)
  }

  setRightLegends(enabled) {
    const custom = this.params.right_legend ? this.params.right_legend : {}
    const defaults = { name: "Y", height: "j", width: "k", hleft: false, wtop: true }
    const props = { ...this.getLegendProps(), ...defaults, ...custom }
    this.right.setLegends(enabled, props, this.getText)
  }

  setResultLegends(enabled) {
    const custom = this.params.result_legend ? this.params.result_legend : {}
    const defaults = { name: "XY", height: "i", width: "k", hleft: false, wtop: false }
    const props = { ...this.getLegendProps(), ...defaults, ...custom }
    this.result.setLegends(enabled, props, this.getText)
  }

  setLegends(enabled) {
    this.params.legends = enabled
    this.setLeftLegends(enabled)
    this.setRightLegends(enabled)
    this.setResultLegends(enabled)
  }
}

//
// Attn (Q @ K^T) @ V
//

export class Attn {
  constructor(params, getText) {
    this.getText = getText
    this.params = { ...params }
    this.group = new THREE.Group()

    this.H = params.n_q
    this.D = params.d_qk + this.params.d_v
    this.W = params.n_kv

    this.initVis()
  }

  initVis(params = undefined) {
    if (params) {
      this.params = { ...params }
    }
    this.group.clear()

    // this._setAbsmax(this.left_data, this.right_data, this.result_data)

    this.initmm1()
    this.initmm2()

    // this.animation = 'none'
    // this.setAnimation(this.params.animation)

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
    const mm1_params = {
      ...this.params, ...{
        I: this.params.n_q,
        J: this.params.d_qk,
        K: this.params.n_kv,
        'left init': this.params['q init'],
        'left sparsity': this.params['q sparsity'],
        left_legend: { name: "Q", height: "n_q", width: "d_qk" },
        'right init': this.params['k^t init'],
        'right sparsity': this.params['k^t sparsity'],
        right_legend: { name: "K^T", height: "d_qk", width: "n_kv" },
        result_legend: { name: "attn", height: "", width: "" },
        epilog: this.params['attn epilog'],
        pos: new THREE.Vector3(0, 0, 0),
      }
    }
    this.mm1 = new MatMul(mm1_params, this.getText)
    this.group.add(this.mm1.group)
  }

  initmm2() {
    if (this.mm2) {
      this.group.remove(this.mm2.group)
    }
    const mm2_params = {
      ...this.params, ...{
        I: this.params.n_q,
        J: this.params.n_kv,
        K: this.params.d_v,
        left: this.mm1.result,
        'right init': this.params['v init'],
        'right sparsity': this.params['v sparsity'],
        right_legend: { name: "V", height: "n_kv", width: "d_v" },
        result_legend: { name: "out", height: "n_q", width: "d_v", wtop: true },
        epilog: this.params['result epilog'],
        right_rot: new THREE.Vector3(0, Math.PI, 0),

        // alternating
        right_pos: new THREE.Vector3(0, -this.H - 1, 0),

        result_rot: new THREE.Vector3(0, Math.PI, 0),
        rot: new THREE.Vector3(0, Math.PI / 2, 0),
        pos: new THREE.Vector3(0, 0, this.mm1.D + 1),


      }
    }
    this.mm2 = new MatMul(mm2_params, this.getText)
    this.group.add(this.mm2.group)
  }

  initQ(params = undefined) {
    if (params) {
      this.params = { ...params }
    }
    this.initmm1()
    this.initVis()
  }

  setNQ(n_q) {
    this.H = this.params.n_q = n_q
    this.mm1.setI(n_q)
    this.mm2.params.left = this.mm1.result
    this.mm2.params.right_pos = new THREE.Vector3(0, -this.H - 1, 0)
    this.mm2.setI(n_q)
    this.setPosition()
  }

  setNKV(n_kv) {
    this.W = this.params.n_kv = n_kv
    this.mm1.setK(n_kv)
    this.mm2.params.left = this.mm1.result
    this.initmm2()
    this.setPosition()
  }

  setAttnEpilog(epilog) {
    this.params['attn epilog'] = epilog
    this.mm1.initResultData()
    this.initVis()
  }

  setResultEpilog(epilog) {
    this.params['result epilog'] = epilog
    this.mm2.initResultData()
    this.initVis()
  }

  setGuides(enabled) {
    this.mm1.setGuides(enabled)
    this.mm2.setGuides(enabled)
  }

  setLegends(enabled) {
    this.mm1.setLegends(enabled)
    this.mm2.setLegends(enabled)
  }

  bump() {
    this.mm1.bump()
    this.mm2.bump()
  }
}

//
// MLP W1 @ (W0 @ X)
//

export class MLP {
  constructor(params, getText) {
    this.getText = getText
    this.params = { ...params }
    this.group = new THREE.Group()

    this.mms = []
    const nlayers = params.nlayers ? params.nlayers : 5
    for (let i = 0; i < nlayers; i++) {
      const I = params[`I_${i}`]
      const J = i == 0 ? params.J_0 : params[`I_${i - 1}`]
      const K = params.K
      const mm_params = {
        I: I,
        J: J,
        K: K,
        ...this.params,
        ...(i > 0 ? { right: this.mms[i - 1].result } : {}),
        ...{
          left_legend: { name: `L${i}`, height: `i${i}`, width: i == 0 ? "j0" : `j${i} = i${i - 1}` },
          right_legend: i == 0 ? { name: `R${i}`, height: `j${i}`, width: "k" } : {},
          result_legend: { name: `R${i + 1} = L${i} R${i}`, height: `i${i}`, width: "k", hleft: i % 2 == 1 },
          pos: i == 0 ?
            new THREE.Vector3(0, 0, 0) :
            (prev =>
              new THREE.Vector3(
                0,
                prev.group.position.y + (i % 2 == 0 ? -prev.D - 1 : 0),
                prev.group.position.z + (i % 2 == 1 ? prev.D + 1 : 0)
              )
            )(this.mms[i - 1])
          ,
        },
        ...{
          left_pos: new THREE.Vector3({ left: 0, right: K + 1, alternating: i % 2 == 1 ? K + 1 : 0 }[params.lhs], 0, 0),
        },
        ...(i % 2 == 0 ? {} : {
          // left_pos: new THREE.Vector3(params.lhs == 'alternating' ? this.mms[0].W + 1 : 0, 0, 0),
          left_rot: new THREE.Vector3(-Math.PI / 2, Math.PI, 0),
          result_pos: new THREE.Vector3(0, -J, -J),
          result_rot: new THREE.Vector3(-Math.PI / 2, 0, 0),
        })
      }

      const mm = new MatMul(mm_params, getText)
      this.mms.push(mm)
      this.group.add(mm.group)
    }

    this.setPosition()
  }

  setPosition() {
    const first_mm = this.mms[0]
    const n = this.mms.length
    const last_mm = this.mms[n - 1]

    this.H = last_mm.group.position.y - (n % 2 == 0 ? last_mm.D : last_mm.H) + first_mm.group.position.y
    this.D = last_mm.group.position.z + (n % 2 == 0 ? last_mm.H : last_mm.D) - first_mm.group.position.z
    this.W = last_mm.group.position.x + last_mm.W - first_mm.group.position.x

    // center cube on 0,0,0 if no pos given
    // note: don't save into params
    const pos = this.params.pos ? this.params.pos :
      new THREE.Vector3(-this.W / 2, -this.H / 2, -this.D / 2)
    console.log(`HEY ${pos.x} ${pos.y} ${pos.z}`)
    this.group.position.x = pos.x
    this.group.position.y = pos.y
    this.group.position.z = pos.z
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
// MLP pytorch style (x @ w0^T) @ w1^T @ ...
//

export class MLPT {
  constructor(params, getText) {
    this.getText = getText
    this.params = { ...params }
    this.group = new THREE.Group()

    this.mms = []
    const nlayers = params.nlayers ? params.nlayers : 5
    for (let i = 0; i < nlayers; i++) {
      const I = params.I
      const J = i == 0 ? params.J_0 : params[`K_${i - 1}`]
      const K = params[`K_${i}`]
      const mm_params = {
        I: I,
        J: J,
        K: K,
        ...this.params,
        ...(i > 0 ? { left: this.mms[i - 1].result } : {}),
        ...{
          left_legend: { name: `L${i}`, height: "i", width: `j${i}` },
          right_legend: { name: `R${i}`, height: i == 0 ? "j0" : `j${i} = k${i - 1}`, width: `k${i} ` },
          result_legend: { name: `L${i + 1} = L${i} R${i}`, height: "i", width: `k${i}`, wtop: i % 2 == 1 },
          pos: i == 0 ?
            new THREE.Vector3(0, 0, 0) :
            (prev =>
              new THREE.Vector3(
                prev.group.position.x - (i % 2 == 0 ? -prev.D - 1 : 0),
                0,
                prev.group.position.z + (i % 2 == 1 ? prev.D + 1 : 0),
              )
            )(this.mms[i - 1])
          ,
          // pos: new THREE.Vector3(0, 0, this.mm1.D + 1),
          // rot: new THREE.Vector3(0, Math.PI / 2, 0),
        },
        ...{
          right_pos: new THREE.Vector3(
            0,
            { up: 0, down: -I - 1, alternating: i % 2 == 1 ? -I - 1 : 0 }[params.rhs],
            0,
          )
        },
        ...(i % 2 == 0 ? {} : {
          right_rot: new THREE.Vector3(0, Math.PI, Math.PI / 2),
          result_pos: new THREE.Vector3(J, 0, -J),
          result_rot: new THREE.Vector3(0, Math.PI / 2, 0),
        })
      }
      const mm = new MatMul(mm_params, getText)
      this.mms.push(mm)
      this.group.add(mm.group)
    }

    this.setPosition()
  }

  setPosition() {
    const first_mm = this.mms[0]
    const n = this.mms.length
    const last_mm = this.mms[n - 1]

    this.H = last_mm.group.position.y + last_mm.H
    this.W = last_mm.group.position.x + (n % 2 == 0 ? last_mm.D : last_mm.W) - first_mm.group.position.x
    this.D = last_mm.group.position.z + (n % 2 == 0 ? last_mm.W : last_mm.D) - first_mm.group.position.z

    // center cube on 0,0,0 if no pos given
    // note: don't save into params
    const pos = this.params.pos ? this.params.pos :
      new THREE.Vector3(-this.W / 2, this.H / 2, -this.D / 2)
    this.group.position.x = pos.x
    this.group.position.y = pos.y
    this.group.position.z = pos.z
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

