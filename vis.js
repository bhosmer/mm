import * as THREE from 'three'
import * as util from './util.js'

//
// shader
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
// initialization
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

const INIT_FUNCS = {
  rows: (i, j, h, w) => h > 1 ? i / (h - 1) : 0,
  cols: (i, j, h, w) => w > 1 ? j / (w - 1) : 0,
  'row major': (i, j, h, w) => h * w > 1 ? (i * w + j) / (h * w - 1) : 0,
  'col major': (i, j, h, w) => h * w > 1 ? (j * h + i) / (h * w) : 0,
  'pt linear': (i, j, h, w) => (2 * Math.random() - 1) / Math.sqrt(w),
  'pt linear+': (i, j, h, w) => Math.max((2 * Math.random() - 1) / Math.sqrt(w)),
  uniform: (i, j, h, w) => Math.random(),
  gaussian: (i, j, h, w) => gaussianRandom(0.5, 0.5),
  tril: (i, j, h, w) => j <= i ? 1 : 0,
  triu: (i, j, h, w) => j >= i ? 1 : 0,
  eye: (i, j, h, w) => i == j ? 1 : 0,
  diff: (i, j, h, w) => i == j ? 1 : i == j + 1 ? -1 : 0,
  ones: (i, j, h, w) => 1,
  zeros: (i, j, h, w) => 0,
}

const USE_RANGE = ['rows', 'cols', 'row major', 'col major', 'uniform', 'gaussian']

function useRange(name) {
  return USE_RANGE.indexOf(name) >= 0
}

function getInitFunc(name, sparsity, base = 0, range = 1) {
  const f = INIT_FUNCS[name]
  if (!f) {
    throw Error(`unrecognized initializer ${name}`)
  }
  const scaled = useRange(name) && (base != 0 || range != 1) ?
    (i, j, h, w) => base + range * f(i, j, h, w) :
    f
  const sparse = sparsity > 0 ?
    (i, j, h, w) => Math.random() > sparsity ? scaled(i, j, h, w) : 0 :
    scaled
  return sparse
}

function softmax_(h, w, data) {
  let ptr = 0
  for (let i = 0; i < h; i++) {
    const rptr = ptr
    let denom = 0
    for (let j = 0; j < w; j++, ptr++) {
      denom += Math.exp(data[ptr])
    }
    for (let j = 0, ptr = rptr; j < w; j++, ptr++) {
      data[ptr] = Math.exp(data[ptr]) / denom
    }
  }
}

function layernorm_(h, w, data) {
  // (x - x.mean()) / torch.sqrt((x ** 2).mean() - x.mean() ** 2)
  const mean = data.reduce((acc, x) => acc + x) / data.length
  const data2 = data.map(x => x ** 2)
  const mean2 = data2.reduce((acc, x) => acc + x) / data2.length
  const denom = Math.sqrt(mean2 - mean ** 2)
  const n = h * w
  for (let ptr = 0; ptr < n; ptr++) {
    const x = data[ptr]
    data[ptr] = (x - mean) / denom
  }
}

function getInPlaceEpilog(name) {
  if (name.startsWith('softmax')) {
    return softmax_
  }
  if (name.startsWith('layernorm')) {
    return layernorm_
  }
  return undefined
}

//
// Array2D
//

function arrayInit(a, h, w, f, epi = undefined) {
  for (let i = 0, ptr = 0; i < h; i++) {
    for (let j = 0; j < w; j++, ptr++) {
      const x = f(i, j, h, w)
      if (isNaN(x)) {
        throw Error(`HEY init f(${i}, ${j}, ${h}, ${w}) is NaN`)
      }
      a[ptr] = x
    }
  }
  const epi_ = epi && getInPlaceEpilog(epi)
  if (epi_) {
    epi_(h, w, a)
  }
}

class Array2D {

  static fromInit(h, w, f, epi = undefined) {
    const data = new Float32Array(h * w)
    arrayInit(data, h, w, f, epi)
    return new Array2D(h, w, data, epi)
  }

  constructor(h, w, data) {
    this.h = h
    this.w = w
    this.data = data
  }

  reinit(f, epi = undefined) {
    arrayInit(this.data, this.h, this.w, f, epi)
  }

  numel() {
    return this.h * this.w
  }

  get(i, j) {
    return this.data[this.addr(i, j)]
  }

  set(i, j, x) {
    if (isNaN(x)) {
      throw Error(`HEY set(${i}, ${j}, ${x})`)
    }
    this.data[this.addr(i, j)] = x
  }

  addr(i, j) {
    return i * this.w + j
  }

  absmax() {
    let x = 0
    const data = this.data
    for (let i = 0, y = Math.abs(data[0]); i < data.length; y = Math.abs(data[++i])) {
      if (x < y) {
        x = y
      }
    }
    return x
  }

  absmin() {
    let x = Infinity
    const data = this.data
    for (let i = 0, y = Math.abs(data[0]); i < data.length; y = Math.abs(data[++i])) {
      if (x > y) {
        x = y
      }
    }
    return x
  }

  transpose() {
    return Array2D.fromInit(this.w, this.h, (i, j, h, w) => this.get(j, i))
  }

  map(f) {
    const data = new Float32Array(n)
    for (let ptr = 0; ptr < n; ptr++) {
      data[ptr] = f(this.data[ptr])
    }
    return new Array2D(this.h, this.w, data)
  }

  map2(f, a) {
    if (a.h != this.h || a.w != this.w) {
      throw Error(`shape error: this ${this.h} ${this.w} a ${a.h} ${a.w}`)
    }
    const n = this.h * this.w
    const data = new Float32Array(n)
    for (let ptr = 0; ptr < n; ptr++) {
      data[ptr] = f(this.data[ptr], a.data[ptr])
    }
    return new Array2D(this.h, this.w, data)
  }

  add(a) {
    return this.map2((x, y) => x + y, a)
  }
}

const ELEM_SIZE = 1792

//
// Mat
//

function emptyPoints(h, w) {
  const n = h * w
  const geom = new THREE.BufferGeometry();
  const points = new Float32Array(n * 3)
  for (let i = 0, ptr3 = 0; i < h; i++) {
    for (let j = 0; j < w; j++, ptr3 += 3) {
      points[ptr3] = j
      points[ptr3 + 1] = i
      points[ptr3 + 2] = 0
    }
  }
  geom.setAttribute('position', new THREE.BufferAttribute(points, 3));
  geom.setAttribute('pointSize', new THREE.Float32BufferAttribute(new Float32Array(n), 1))
  geom.setAttribute('pointColor', new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3))
  return new THREE.Points(geom, MATERIAL)
}

export class Mat {

  static fromInit(h, w, init, container) {
    return new Mat(Array2D.fromInit(h, w, init), container)
  }

  static dataFromParams(h, w, params) {
    const init_base = params['init min']
    const init_range = Math.max(0, params['init max'] - params['init min'])
    const init_name = params['left init']
    if (!init_name) {
      throw Error(`no initializer specified at params['left_init']`)
    }
    const sparsity = params['left sparsity']
    const init = getInitFunc(init_name, sparsity, init_base, init_range)
    return Array2D.fromInit(h, w, init)
  }

  static fromParams(h, w, params, getText) {
    let data
    if (params.data) {
      if (params.data.h != h || params.data.w != w) {
        throw Error(`shape mismatch: h ${h} w ${w} params.data.h ${params.data.h} w ${params.data.w}`)
      }
      data = params.data
    } else {
      data = Mat.dataFromParams(h, w, params)
    }

    const m = new Mat(data, undefined, params)
    m.setGuides(params.guides)

    const custom = params.legend ? params.legend : {}
    const defaults = { name: "X", height: "i", width: "j", hleft: true, wtop: false }
    const props = { ...m.getLegendProps(), ...defaults, ...custom }
    m.setLegends(params.legends, props, getText)

    return m
  }

  constructor(data, container, params) {
    if (container) {
      if (params) {
        throw Error('passed both container and params to Mat')
      }
      this.params = { ...container.params }
      this.getGlobalAbsmax = () => container.getGlobalAbsmax()
    } else {
      if (!params) {
        throw Error('passed neither container nor params to Mat')
      }
      this.params = { ...params }
      this.getGlobalAbsmax = () => this.absmax
    }

    this.h = data.h
    this.w = data.w
    this.points = emptyPoints(this.h, this.w)

    this.data = data
    this.initVis()

    this.group = new THREE.Group()
    this.group.add(this.points)
  }

  initVis() {
    this.absmax = this.data.absmax()
    this.absmin = this.data.absmin()
    const sizes = this.getPointSizes()
    const colors = this.getPointColors()
    for (let i = 0, ptr = 0; i < this.h; i++) {
      for (let j = 0; j < this.w; j++, ptr++) {
        const x = this.data.data[ptr]
        sizes[ptr] = this.sizeFromData(x)
        this.setElemHSL(colors, ptr, x)
      }
    }
    this.points.geometry.attributes.pointSize.needsUpdate = true
    this.points.geometry.attributes.pointColor.needsUpdate = true
  }

  reinit(f, epi = undefined) {
    this.data.reinit(f, epi)
    this.initVis()
  }

  getPointSizes() {
    return this.points.geometry.attributes.pointSize.array
  }

  getPointColors() {
    return this.points.geometry.attributes.pointColor.array
  }

  getAbsmax() {
    return this.absmax
  }

  sizeFromData(x) {
    if (x == undefined) {
      throw Error(`HEY sizeFromData(${x})`)
    }
    if (isNaN(x)) {
      return 0
    }

    const local_sens = this.params.sensitivity == 'local'
    const absx = Math.abs(x)
    const [min, max] = local_sens ? [this.absmin, this.absmax] : [0, this.getGlobalAbsmax()]

    const vol = min == max ? 1 : (absx - min) / (max - min)
    const zsize = this.params['zero size'] * ELEM_SIZE
    const size = zsize + (ELEM_SIZE - zsize) * Math.cbrt(vol)

    if (size < 0 || size > ELEM_SIZE * 1.1 || isNaN(size)) {
      throw Error(`HEY size ${size} absx ${absx} max ${max} min ${min} zsize ${zsize} sens ${local_sens}`)
    }

    return size
  }

  colorFromData(x) {
    if (x == undefined) {
      throw Error(`HEY colorFromData(${x})`)
    }
    if (isNaN(x)) {
      return new THREE.Color().setHSL(0, 0, 0)
    }

    const local_sens = this.params.sensitivity == 'local'
    const absx = Math.abs(x)
    const [min, max] = local_sens ? [this.absmin, this.absmax] : [0, this.getGlobalAbsmax()]

    const hue_vol = min == max ? x : x / (max - min)
    const gap = this.params['hue gap'] * Math.sign(x)
    const hue = (this.params['zero hue'] + gap + (Math.cbrt(hue_vol) * this.params['hue spread'])) % 1

    const light_vol = min == max ? 1 : (absx - min) / (max - min)
    const range = this.params['max light'] - this.params['zero light']
    const light = this.params['zero light'] + range * Math.cbrt(light_vol)

    return new THREE.Color().setHSL(hue, 1.0, light)
  }

  setElemHSL(a, i, x) {
    this.colorFromData(x).toArray(a, i * 3)
  }

  setSize(i, j, x) {
    this.points.geometry.attributes.pointSize.array[this.data.addr(i, j)] = x
    this.points.geometry.attributes.pointSize.needsUpdate = true
  }

  setHSL(i, j, x) {
    this.setElemHSL(this.points.geometry.attributes.pointColor.array, this.data.addr(i, j), x)
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

  setData(i, j, x, adjust_minmax = false) {
    if (isNaN(x)) {
      throw Error(`HEY setData(${i}, ${j}, ${x})`)
    }
    this.data.set(i, j, x)
    const absx = Math.abs(x)
    if (absx >= this.absmin && absx <= this.absmax) {
      this.setSize(i, j, this.sizeFromData(x))
      this.setHSL(i, j, x)
    } else if (adjust_minmax) {
      this.initVis()
    } else {
      throw Error(`HEY setData(${i}, ${j}, ${x}) this.absmax ${this.absmax} this.absmin ${this.absmin}`)
    }
  }

  show(i, j) {
    this.setSize(i, j, this.sizeFromData(this.getData(i, j)))
    this.setHSL(i, j, this.getData(i, j))
  }

  showAll() {
    for (let i = 0; i < this.h; i++) {
      for (let j = 0; j < this.w; j++) {
        this.show(i, j)
      }
    }
  }

  hideAll() {
    for (let i = 0; i < this.h; i++) {
      for (let j = 0; j < this.w; j++) {
        this.setSize(i, j, this.sizeFromData(NaN))
        this.setHSL(i, j, NaN)
      }
    }
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

  getLegendProps() {
    const custom = this.params.legend_props ? this.params.legend_props : {}
    const sa_geo = Math.cbrt(Math.max(5, this.h) * Math.max(this.w, 5))
    const defaults = {
      name_color: 0xccccff,
      name_size: sa_geo / 2,
      dim_color: 0x00aaff,
      dim_size: sa_geo / 5,
    }
    const res = { ...defaults, ...custom }
    return res
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

  constructor(params, getText, container = undefined) {
    this.getText = getText
    this.params = { ...params }
    this.group = new THREE.Group()
    this.getGlobalAbsmax = container ? () => container.getGlobalAbsmax() : () => this.getAbsmax()

    this.H = params.I
    this.D = params.J
    this.W = params.K

    this.init_base = this.params['init min']
    this.init_range = Math.max(0, this.params['init max'] - this.params['init min'])
    this.initLeftData()
    this.initRightData()
    this.initResultData()

    this.initVis()
  }

  initLeftData() {
    if (this.params.left) {
      this.left_data = this.params.left.data
      return
    }
    if (this.params.left_data) {
      this.left_data = this.params.left_data
      return
    }
    const init = this.params['left init']
    const sparsity = this.params['left sparsity']
    const left_init = getInitFunc(init, sparsity, this.init_base, this.init_range)
    this.left_data = Array2D.fromInit(this.H, this.D, left_init)
  }

  initRightData() {
    if (this.params.right) {
      this.right_data = this.params.right.data
      return
    }
    if (this.params.right_data) {
      this.right_data = this.params.right_data
      return
    }
    const init = this.params['right init']
    const sparsity = this.params['right sparsity']
    const right_init = getInitFunc(init, sparsity, this.init_base, this.init_range)
    this.right_data = Array2D.fromInit(this.D, this.W, right_init)
  }

  initResultData() {
    if (this.params.result) {
      throw Error(`HEY this.params.result`)
      this.result_data = this.params.result.data
      return
    }
    if (this.params.result_data) {
      throw Error(`HEY this.params.result_data`)
      this.result_data = this.params.result_data
      return
    }
    const result_init = (y, x, h, w) => this._result_val(this.left_data, this.right_data, y, x)
    this.result_data = Array2D.fromInit(this.H, this.W, result_init, this.params.epilog)
  }

  _result_val(a, b, i, k, minj = undefined, maxj = undefined) {
    let x = 0.0
    if (minj == undefined) {
      minj = 0
    }
    if (maxj == undefined) {
      maxj = a.w
    }
    for (let j = minj; j < maxj; j++) {
      x += a.get(i, j) * b.get(j, k)
    }
    const epi = this.params.epilog
    return epi == 'x/J' ? x / this.D :
      epi == 'x/sqrt(J)' || epi == 'softmax(x/sqrt(J))' ? x / Math.sqrt(this.D) :
        epi == 'tanh(x)' ? Math.tanh(x) :
          epi == 'relu(x)' ? Math.max(0, x) :
            x
  }

  result_val(i, k, minj = undefined, maxj = undefined) {
    return this._result_val(this.left.data, this.right.data, i, k, minj, maxj)
  }

  initVis(params = undefined) {
    if (params) {
      this.params = { ...params }
    }
    this.group.clear()

    this.initAbsmax()
    this.initLeftVis()
    this.initRightVis()
    this.initResultVis()

    this.setAnimation()

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
    this.left = new Mat(this.left_data, this)
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
    this.right = new Mat(this.right_data, this)
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
    if (this.params.result) {
      throw Error(`HEY this.params.result`)
      this.result = this.params.result
      return
    }
    if (this.result) {
      this.group.remove(this.result.group)
    }
    this.result = new Mat(this.result_data, this)
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

  initAbsmax() {
    this.absmax = Math.max(this.left_data.absmax(), this.right_data.absmax(), this.result_data.absmax())
  }

  getAbsmax() {
    return this.absmax
  }

  setAnimation() {
    const prev_alg = this.alg
    this.alg = this.params.alg

    if (this.params.alg == 'none') {
      this.left.showAll()
      this.right.showAll()
      this.result.showAll()
    } else {
      if (this.params['hide inputs']) {
        this.left.hideAll()
        this.right.hideAll()
      }
      if (this.params['j threads'] > 1) {
        this.result.hideAll()
      }
    }

    if (this.alg == 'dotprod') {
      this.initAnimDotprod()
    } else if (this.alg == 'axpy') {
      this.initAnimAXPY()
    } else if (this.alg == 'mvprod') {
      this.result.hideAll()
      const mvprod_init = (i, j, h, w) => this.dotprod_val(i, j, 0)
      this.mvprod = Mat.fromInit(this.H, this.D, mvprod_init, this)
      this.mvprod.points.rotation.y = Math.PI / 2
      this.mvprod.points.rotation.z = Math.PI
      this.group.add(this.mvprod.points)
      this.bump = this.bump_mvprod
      this.curk = this.W - 1
    } else if (this.alg == 'vmprod') {
      this.result.hideAll()
      const vmprod_init = (i, j, h, w) => this.dotprod_val(0, i, j)
      this.vmprod = Mat.fromInit(this.D, this.W, vmprod_init, this)
      this.vmprod.points.rotation.x = Math.PI / 2
      this.group.add(this.vmprod.points)
      this.bump = this.bump_vmprod
      this.curi = this.H - 1
    } else if (this.alg == 'vvprod') {
      this.result.hideAll()
      const vvprod_init = (i, j, h, w) => this.dotprod_val(i, 0, j)
      this.vvprod = Mat.fromInit(this.H, this.W, vvprod_init, this)
      this.vvprod.points.rotation.x = Math.PI
      this.group.add(this.vvprod.points)
      this.bump = this.bump_vvprod
      this.curj = this.D - 1
    } else if (this.alg == 'none') {
      if (prev_alg == 'vv_prod' || prev_alg == 'axpy') {
        this.initResultData() // depthwise animations accum into result
        this.initResultVis()
      } else {
        this.result.showAll()
      }
    }
  }

  dotprod_val(i, j, k) {
    return this.left.getData(i, j) * this.right.getData(j, k)
  }

  getThreadInfo() {
    const nh = Math.min(this.params['i threads'], this.H)
    const hp = Math.floor(this.H / nh)
    const nd = Math.min(this.params['j threads'], this.D)
    const dp = Math.floor(this.D / nd)
    const nv = Math.min(this.params['k threads'], this.W)
    const vp = Math.floor(this.W / nv)
    return { nh, hp, nd, dp, nv, vp }
  }

  initAnimDotprod() {
    this.dotprods = []
    this.dpgroup = new THREE.Group()
    this.dpresults = []
    const { nh, hp, nd, dp, nv, vp } = this.getThreadInfo()

    for (let hi = 0; hi < nh; hi++) {
      for (let vi = 0; vi < nv; vi++) {
        const dotprod_init = (i, j, h, w) => this.dotprod_val(hi * hp, j, vi * vp)
        const dotprod = Mat.fromInit(1, this.D, dotprod_init, this)
        dotprod.points.rotation.y = -Math.PI / 2
        dotprod.points.position.y -= hi * hp
        dotprod.points.position.x += vi * vp
        this.dotprods.push(dotprod)
        this.dpgroup.add(dotprod.points)
      }
    }
    this.group.add(this.dpgroup)

    if (nd > 1) {
      for (let di = 0; di < nd; di++) {
        const dpresult_init = (y, x, h, w) => this.result_val(y, x, di * dp, (di + 1) * dp)
        const dpresult = Mat.fromInit(this.H, this.W, dpresult_init, this)
        dpresult.group.position.z = di * dp + dp - 1
        dpresult.group.rotation.x = Math.PI
        this.dpresults.push(dpresult)
        this.group.add(dpresult.group)
        dpresult.hideAll()
      }
    }

    let curi = hp - 1
    let curk = vp - 1

    this.bump = () => {
      const [oldi, oldk] = [curi, curk]
      if (oldk < vp - 1) {
        curk += 1
      } else {
        curk = 0
        curi = oldi < hp - 1 ? curi + 1 : 0
      }

      // update result faces
      if (nd == 1) {
        if (curi == 0 && curk == 0) {
          this.result.hideAll()
        }
        for (let hi = 0; hi < nh; hi++) {
          for (let vi = 0; vi < nv; vi++) {
            this.result.show(hi * hp + curi, vi * vp + curk)
          }
        }
      } else {
        if (curi == 0 && curk == 0) {
          this.dpresults.forEach(dpresult => dpresult.hideAll())
        }
        for (let hi = 0; hi < nh; hi++) {
          for (let vi = 0; vi < nv; vi++) {
            this.dpresults.forEach(dpresult => {
              dpresult.show(hi * hp + curi, vi * vp + curk)
            })
          }
        }
      }

      if (!this.params['hide inputs']) {
        // hilight operand row/cols
        if (oldk != curk) {
          for (let vi = 0; vi < nv; vi++) {
            this.right.bumpColumnColor(oldk + vi * vp, false)
            this.right.bumpColumnColor(curk + vi * vp, true)
          }
        }
        if (oldi != curi) {
          for (let hi = 0; hi < nh; hi++) {
            this.left.bumpRowColor(oldi + hi * hp, false)
            this.left.bumpRowColor(curi + hi * hp, true)
          }
        }
      }

      // move and recolor dot product vectors
      this.dpgroup.position.x = curk
      this.dpgroup.position.y = -curi
      for (let hi = 0, ptr = 0; hi < nh; hi++) {
        for (let vi = 0; vi < nv; vi++, ptr++) {
          const dotprod = this.dotprods[ptr]
          dotprod.reinit((i, j, h, w) => this.dotprod_val(hi * hp + curi, j, vi * vp + curk))
        }
      }
    }
  }

  initAnimAXPY() {
    // TODO
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
    for (let j = 0; j < this.D; j++) {
      for (let k = 0; k < this.W; k++) {
        this.vmprod.setData(j, k, this.dotprod_val(i, j, k))
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
    for (let i = 0; i < this.H; i++) {
      for (let j = 0; j < this.D; j++) {
        this.mvprod.setData(i, j, this.dotprod_val(i, j, k))
      }
    }
  }

  bump_vvprod() {
    const oldj = this.curj

    if (oldj < this.D - 1) {
      this.curj += 1
    } else {
      this.curj = 0
    }

    const j = this.curj

    // update result face
    for (let i = 0; i < this.H; i++) {
      for (let k = 0; k < this.W; k++) {
        this.result.setData(i, k, this.result_val(i, k, 0, j + 1))
      }
    }

    this.left.bumpColumnColor(oldj, false)
    this.left.bumpColumnColor(j, true)
    this.right.bumpRowColor(oldj, false)
    this.right.bumpRowColor(j, true)

    // move and recolor dot product vector
    this.vvprod.points.position.z = this.left.points.geometry.attributes.position.array[j * 3]
    for (let i = 0; i < this.H; i++) {
      for (let k = 0; k < this.W; k++) {
        this.vvprod.setData(i, k, this.dotprod_val(i, j, k))
      }
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

  setLeftLegends(enabled) {
    const custom = this.params.left_legend ? this.params.left_legend : {}
    const defaults = { name: "X", height: "i", width: "j", hleft: true, wtop: false }
    const props = { ...this.left.getLegendProps(), ...defaults, ...custom }
    this.left.setLegends(enabled, props, this.getText)
  }

  setRightLegends(enabled) {
    const custom = this.params.right_legend ? this.params.right_legend : {}
    const defaults = { name: "Y", height: "j", width: "k", hleft: false, wtop: true }
    const props = { ...this.right.getLegendProps(), ...defaults, ...custom }
    this.right.setLegends(enabled, props, this.getText)
  }

  setResultLegends(enabled) {
    const custom = this.params.result_legend ? this.params.result_legend : {}
    const defaults = { name: "XY", height: "i", width: "k", hleft: false, wtop: false }
    const props = { ...this.result.getLegendProps(), ...defaults, ...custom }
    this.result.setLegends(enabled, props, this.getText)
  }

  setLegends(enabled) {
    this.params.legends = enabled
    if (!this.params.left) {
      this.setLeftLegends(enabled)
    }
    if (!this.params.right) {
      this.setRightLegends(enabled)
    }
    if (!this.params.result) {
      this.setResultLegends(enabled)
    }
  }

  getGuiCallback(name) {
    throw Error(`HEY unknown callback ${name}`)
  }
}

//
// Attn (Q @ K.T) @ V
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
    this.initAbsmax()

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
      ...this.params,
      I: this.params.n_q,
      J: this.params.d_qk,
      K: this.params.n_q,
      'left init': this.params['q init'],
      'left sparsity': this.params['q sparsity'],
      left_legend: { name: "Q", height: "n_q", width: "d_qk" },
      'right init': this.params['k^t init'],
      'right sparsity': this.params['k^t sparsity'],
      right_legend: { name: "K.T", height: "d_qk", width: "n_q" },
      result_legend: { name: "attn", height: "", width: "" },
      epilog: this.params['attn epilog'],
      pos: new THREE.Vector3(0, 0, 0),
    }
    this.mm1 = new MatMul(mm1_params, this.getText)
    this.group.add(this.mm1.group)
  }

  initmm2() {
    if (this.mm2) {
      this.group.remove(this.mm2.group)
    }
    const mm2_params = {
      ...this.params,
      I: this.params.n_q,
      J: this.params.n_q,
      K: this.params.d_v,
      left: this.mm1.result,
      'right init': this.params['v init'],
      'right sparsity': this.params['v sparsity'],
      right_legend: { name: "V", height: "n_q", width: "d_v" },
      result_legend: { name: "out", height: "n_q", width: "d_v", wtop: true },
      epilog: this.params['result epilog'],
      right_rot: new THREE.Vector3(0, Math.PI, 0),

      // alternating
      right_pos: new THREE.Vector3(0, -this.H - 1, 0),

      result_rot: new THREE.Vector3(0, Math.PI, 0),
      rot: new THREE.Vector3(0, Math.PI / 2, 0),
      pos: new THREE.Vector3(0, 0, this.mm1.D + 1),

    }
    this.mm2 = new MatMul(mm2_params, this.getText)
    this.group.add(this.mm2.group)
  }

  initAbsmax() {
    this.absmax = Math.max(this.mm1.getAbsmax(), this.mm2.getAbsmax())
  }

  getGlobalAbsmax() {
    return this.absmax
  }

  initQ(params = undefined) {
    if (params) {
      this.params = { ...params }
    }
    this.initmm1()
    this.initVis()
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
        ...this.params,

        I: I,
        J: J,
        K: K,

        ...(i > 0 ? { right: this.mms[i - 1].result } : {}),

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
          )(this.mms[i - 1]),

        left_pos: new THREE.Vector3({ left: 0, right: K + 1, alternating: i % 2 == 1 ? K + 1 : 0 }[params.lhs], 0, 0),

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
// MLP pytorch style (x @ w0.T) @ w1.T @ ...
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
        ...this.params,
        I: I,
        J: J,
        K: K,

        ...(i > 0 ? { left: this.mms[i - 1].result } : {}),

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
          )(this.mms[i - 1]),

        right_pos: new THREE.Vector3(
          0,
          { up: 0, down: -I - 1, alternating: i % 2 == 1 ? -I - 1 : 0 }[params.rhs],
          0,
        ),

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

//
// Attn (Q @ K.T) @ V
//

export class Attn2 {
  constructor(params, getText) {
    this.getText = getText
    this.params = { ...params }
    this.group = new THREE.Group()

    this.H = params.n_q
    this.D = params.d_qk + this.params.d_v
    this.W = params.n_q

    this.initVis()
  }

  initVis(params = undefined) {
    if (params) {
      this.params = { ...params }
    }
    this.group.clear()

    this.initqmm()
    this.initkmm()
    this.initmm1()

    this.initvmm()
    this.initmm2()
    this.initomm()

    this.setPosition()
  }

  setPosition() {
    const pos = this.params.pos ? this.params.pos :
      new THREE.Vector3(-this.W / 2, this.H / 2, -this.D / 2)
    this.group.position.x = pos.x
    this.group.position.y = pos.y
    this.group.position.z = pos.z
  }

  initqmm() {
    const qmm_params = {
      ...this.params,

      I: this.params.n_q,
      J: this.params.d_model,
      K: this.params.d_qk,

      ...(this.params.input_data ? {
        left_data: this.params.input_data
      } : {
        'left init': this.params['input init'],
        'left sparsity': this.params['q sparsity'], // TODO        
      }),
      left_legend: { name: "input", height: "n_q", width: "d_model", hleft: false, wtop: true },

      'right init': this.params['wQ init'],
      'right sparsity': this.params['k^t sparsity'], // TODO
      right_legend: { name: "wQ", height: "d_model", width: "d_qk", wtop: false },
      right_pos: new THREE.Vector3(0, -this.params.n_q - 1, 0),

      result_legend: { name: "Q", height: "n_q", width: "d_qk" },
      result_pos: new THREE.Vector3(0, 0, -this.params.d_model - 1),
      // epilog: 'relu(x)', // TODO

      pos: new THREE.Vector3(-2, 0, 0),
      rot: new THREE.Vector3(0, -Math.PI / 2, 0)
    }
    this.qmm = new MatMul(qmm_params, this.getText)
    this.group.add(this.qmm.group)
    this.input_data = this.qmm.left_data // NOTE
  }

  initkmm() {
    const kmm_params = {
      ...this.params,

      I: this.params.d_qk,
      J: this.params.d_model,
      K: this.params.n_q,

      'left init': this.params['wK.T init'],
      'left sparsity': this.params['q sparsity'], // TODO
      left_legend: { name: "wK.T", height: "d_qk", width: "d_model", hleft: false },
      left_pos: new THREE.Vector3(this.params.n_q + 1, 0, 0),

      right_legend: { name: "input.T", height: "d_model", width: "n_q", hleft: true, wtop: false },
      right_data: this.input_data.transpose(),

      result_legend: { name: "K.T", height: "d_k", width: "n_q" },
      result_pos: new THREE.Vector3(0, 0, -this.params.d_model - 1),
      // epilog: 'relu(x)', // TODO

      pos: new THREE.Vector3(0, 2, 0),
      rot: new THREE.Vector3(-Math.PI / 2, 0, 0)
    }
    this.kmm = new MatMul(kmm_params, this.getText)
    this.group.add(this.kmm.group)
  }

  initmm1() {
    const mm1_params = {
      ...this.params,
      I: this.params.n_q,
      J: this.params.d_qk,
      K: this.params.n_q,

      left: this.qmm.result,

      right: this.kmm.result,

      result_legend: { name: "attn", height: "", width: "" },
      epilog: this.params['attn epilog'],

      pos: new THREE.Vector3(0, 0, 0),
    }
    this.mm1 = new MatMul(mm1_params, this.getText)
    this.group.add(this.mm1.group)
  }

  initvmm() {
    const vmm_params = {
      ...this.params,
      I: this.params.n_q,
      J: this.params.d_model,
      K: this.params.d_v,

      left_legend: { name: "input", height: "n_q", width: "d_model", hleft: false },
      left_data: this.input_data, // NOTE
      left_pos: new THREE.Vector3(this.params.d_v + 1, 0, 0),

      'right init': this.params['wV init'],
      'right sparsity': this.params['wV sparsity'],
      right_legend: { name: "wV", height: "d_model", width: "d_v", wtop: false, hleft: true },

      result_legend: { name: "V", height: "n_q", width: "d_v" },
      result_pos: new THREE.Vector3(0, 0, -this.params.d_model - 1),
      // epilog: 'relu(x)', // TODO

      pos: new THREE.Vector3(0, -this.params.n_q - 1, this.params.d_qk + 1),
      rot: new THREE.Vector3(Math.PI / 2, 0, Math.PI / 2)
    }
    this.vmm = new MatMul(vmm_params, this.getText)
    this.group.add(this.vmm.group)
  }

  initmm2() {
    const mm2_params = {
      ...this.params,
      I: this.params.n_q,
      J: this.params.n_q,
      K: this.params.d_v,

      left: this.mm1.result,

      right: this.vmm.result,
      right_rot: new THREE.Vector3(0, Math.PI, 0),
      right_pos: new THREE.Vector3(0, -this.H - 1, 0),

      result_legend: { name: "attn @ V", height: "n_q", width: "d_v", wtop: true },
      epilog: 'none', // this.params['result epilog'],
      result_rot: new THREE.Vector3(0, Math.PI, 0),

      rot: new THREE.Vector3(0, Math.PI / 2, 0),
      pos: new THREE.Vector3(0, 0, this.mm1.D + 1),
    }
    this.mm2 = new MatMul(mm2_params, this.getText)
    this.group.add(this.mm2.group)
  }

  initomm() {
    const omm_params = {
      ...this.params,
      I: this.params.n_q,
      J: this.params.d_v,
      K: this.params.d_model,

      left: this.mm2.result,

      'right init': this.params['wO init'],
      'right sparsity': this.params['w0 sparsity'],
      right_legend: { name: "wO", height: "d_v", width: "d_model" },

      result_legend: { name: "out", height: "n_q", width: "d_model" },
      epilog: this.params['result epilog'],

      pos: new THREE.Vector3(this.params.n_q + 1, 0, this.params.d_qk + 1),
      // rot: new THREE.Vector3(Math.PI / 2, 0, Math.PI / 2)
    }
    this.omm = new MatMul(omm_params, this.getText)
    this.group.add(this.omm.group)
  }
}

export class Attn3 {
  constructor(params, getText) {
    this.getText = getText
    this.params = { ...params }
    this.group = new THREE.Group()

    this.H = params.n_q
    this.D = params.d_qk + params.d_v
    this.W = params.d_model

    this.initVis()
  }

  initVis(params = undefined) {
    if (params) {
      this.params = { ...params }
    }
    this.group.clear()

    const num_heads = this.params.num_heads
    const show_heads = !this.params['hide attn heads']

    const zspace = 4 * Math.sqrt(this.D) * this.params['head spacing']

    // shared input
    const input_params = {
      ...this.params,
      'left init': this.params['input init'],
      'left sparsity': this.params['q sparsity'], // TODO
      legend: { name: "input", height: "n_q", width: "d_model", hleft: false, wtop: true },
    }
    const input = Mat.fromParams(this.params.n_q, this.params.d_model, input_params, this.getText)
    input.group.rotation.x = Math.PI
    input.group.position.x = -this.params.d_model / 2
    input.group.position.y = this.params.n_q / 2
    input.group.position.z = -zspace
    this.group.add(input.group)
    this.params.input_data = input.data

    let output_data = input.data
    for (let i = 0; i < num_heads; i++) {
      const a = new Attn2(this.params, this.getText)
      output_data = output_data.add(a.omm.result.data)
      if (show_heads) {
        a.group.position.z = i * (this.D + zspace)
        if (this.params['hide copied inputs']) {
          a.qmm.left.hideAll()
          a.kmm.right.hideAll()
          a.vmm.left.hideAll()
        }
        this.group.add(a.group)
      }
    }

    // summed output w/residual
    const output_params = {
      ...this.params,
      data: output_data,
      legend: { name: "output", height: "n_q", width: "d_model", hleft: false, wtop: true },
    }
    const output = Mat.fromParams(this.params.n_q, this.params.d_model, output_params, this.getText)
    output.group.rotation.x = Math.PI
    output.group.position.x = -this.params.d_model / 2
    output.group.position.y = this.params.n_q / 2
    output.group.position.z = (this.D + zspace) * num_heads
    this.group.add(output.group)

    //
    this.group.position.z = show_heads ? -(zspace * (num_heads - 1) + this.D * num_heads) / 2 : 0
  }
}
