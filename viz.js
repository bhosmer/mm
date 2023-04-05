import * as THREE from 'three'
import * as util from './util.js'

//
// shader
//

const TEXTURE = new THREE.TextureLoader().load('ball.png')

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

// https://github.com/facebookresearch/shumai/blob/main/test/gradient.test.ts#L5
function sampleSphere(args) {
  const u = sm.randn(args)
  const d = sm.sum(u.mul(u)).sqrt()
  return u.div(d)
}

export const INIT_FUNCS = {
  rows: (i, j, h) => h > 1 ? i / (h - 1) : 0,
  cols: (i, j, h, w) => w > 1 ? j / (w - 1) : 0,
  'row major': (i, j, h, w) => h * w > 1 ? (i * w + j) / (h * w - 1) : 0,
  'col major': (i, j, h, w) => h * w > 1 ? (j * h + i) / (h * w) : 0,
  'pt linear': (i, j, h, w) => (2 * Math.random() - 1) / Math.sqrt(w),
  uniform: () => Math.random(),
  gaussian: () => gaussianRandom(0.5, 0.5),
  // sphere: (i, j, h, w) => sampleSphere([h, w]),
  'tril mask': (i, j) => j <= i ? 1 : 0,
  'triu mask': (i, j) => j >= i ? 1 : 0,
  eye: (i, j) => i == j ? 1 : 0,
  diff: (i, j) => i == j ? 1 : i == j + 1 ? -1 : 0,
}

export const INITS = Object.keys(INIT_FUNCS)

const USE_RANGE = ['rows', 'cols', 'row major', 'col major', 'uniform', 'gaussian']
const USE_DROPOUT = USE_RANGE.concat(['pt linear'])

export const useRange = name => USE_RANGE.indexOf(name) >= 0
export const useDropout = name => USE_DROPOUT.indexOf(name) >= 0

function getInitFunc(name, min = 0, max = 1, dropout = 0) {
  const f = INIT_FUNCS[name]
  if (!f) {
    throw Error(`unrecognized initializer ${name}`)
  }
  const scaled = useRange(name) && (min != 0 || max != 1) ?
    (i, j, h, w) => min + Math.max(0, max - min) * f(i, j, h, w) :
    f
  const sparse = dropout > 0 ?
    (i, j, h, w) => Math.random() > dropout ? scaled(i, j, h, w) : 0 :
    scaled
  return sparse
}

function initFuncFromParams(init_params) {
  const { name, min, max, dropout } = init_params
  return getInitFunc(name, min, max, dropout)
}

// epilogs

export const EPILOGS = [
  'none',
  'x/j',
  'x/sqrt(j)',
  'softmax(x/sqrt(j))',
  'tanh',
  'relu',
  'layernorm',
]

function softmax_(h, w, data) {

  const calc_denom = ptr => {
    let d = 0
    for (let j = 0; j < w; j++, ptr++) {
      d += Math.exp(data[ptr])
      if (!isFinite(d)) {
        console.log(`HEY denom at data[${ptr}) = ${data[ptr]} becomes infinite`)
        break
      }
    }
    return d
  }

  for (let i = 0, ptr = 0; i < h; i++) {
    const denom = calc_denom(ptr)
    for (let j = 0; j < w; j++, ptr++) {
      const x = Math.exp(data[ptr]) / denom
      if (isNaN(x)) {
        console.log(`HEY Math.exp(data[${ptr}) = ${data[ptr]}]) / ${denom} is NaN`)
        data[ptr] = 0
      } else {
        data[ptr] = x
      }
    }
  }
}

function layernorm_(h, w, data) {
  const mean = data.reduce((acc, x) => acc + x) / data.length
  const mean2 = data.map(x => x ** 2).reduce((acc, x) => acc + x) / data.length
  const variance = mean2 - mean ** 2
  const denom = Math.sqrt(variance + 1e-5)
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

function toRange(x, n) {
  return x === undefined ? [0, n] : x.constructor === Array ? x : [x, x + 1]
}

function initArrayData_(data, h, w, init, epi = undefined, r = undefined, c = undefined) {
  const [rstart, rend] = toRange(r, h)
  const [cstart, cend] = toRange(c, w)
  for (let i = rstart; i < rend; i++) {
    for (let j = cstart, ptr = i * w + cstart; j < cend; j++, ptr++) {
      data[ptr] = init(i, j, h, w)
    }
  }
  const epi_ = epi && getInPlaceEpilog(epi)
  if (epi_) {
    epi_(h, w, data)
  }
}

class Array2D {

  static fromInit(h, w, init, epi = undefined) {
    const data = new Float32Array(h * w)
    initArrayData_(data, h, w, init, epi)
    return new Array2D(h, w, data)
  }

  constructor(h, w, data) {
    this.h = h
    this.w = w
    this.data = data
  }

  reinit(f, epi = undefined, r = undefined, c = undefined) {
    initArrayData_(this.data, this.h, this.w, f, epi, r, c)
  }

  numel() {
    return this.h * this.w
  }

  get(i, j) {
    return this.data[this.addr(i, j)]
  }

  addr(i, j) {
    return i * this.w + j
  }

  absmax() {
    const data = this.data
    let absmax = 0
    for (let i = 0; i < data.length; i++) {
      const absx = Math.abs(data[i])
      if (absmax < absx) {
        absmax = absx
      }
    }
    return absmax
  }

  absmin() {
    const data = this.data
    let absmin = Infinity
    for (let i = 0; i < data.length; i++) {
      const absx = Math.abs(data[i])
      if (!isFinite(absmin) || absx < absmin) {
        absmin = absx
      }
    }
    return absmin
  }

  transpose() {
    return Array2D.fromInit(this.w, this.h, (i, j) => this.get(j, i))
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

//
//
//

function grid(info, dims, f) {
  const infos = Array.from(dims).map(d => info[d])
  const loop = (args, infos, f) => infos.length == 0 ?
    f(...args) :
    [...Array(infos[0].n).keys()].map(index => {
      const { size, max } = infos[0]
      const start = index * size
      if (start < max) {  // dead final block when size * n - max > size
        const end = Math.min(start + size, max)
        const extent = end - start
        loop([...args, { index, start, end, extent }], infos.slice(1), f)
      }
    })
  loop([], infos, f)
}

//
// Mat
//

const ELEM_SIZE = 2000
const ZERO_COLOR = new THREE.Color(0, 0, 0)
const COLOR_TEMP = new THREE.Color()

function emptyPoints(h, w, info) {
  const { i: { size: si }, j: { size: sj } } = info
  const n = h * w
  const points = new Float32Array(n * 3)
  for (let i = 0, ptr = 0; i < h; i++) {
    const ioff = Math.floor(i / si)
    for (let j = 0; j < w; j++) {
      const joff = Math.floor(j / sj)
      points[ptr++] = j + joff
      points[ptr++] = i + ioff
      points[ptr++] = 0
    }
  }
  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(points, 3))
  geom.setAttribute('pointSize', new THREE.Float32BufferAttribute(new Float32Array(n), 1))
  geom.setAttribute('pointColor', new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3))
  return new THREE.Points(geom, MATERIAL)
}

export class Mat {

  constructor(data, params, context, init_viz) {
    this.params = params
    this.context = context

    this.data = data
    this.H = data.h
    this.W = data.w
    this.absmax = this.data.absmax()
    this.absmin = this.data.absmin()

    if (init_viz) {
      this.initViz()
    }
  }

  getBlockInfo() {
    const ni = Math.min(this.params.anim['i blocks'], this.H)
    const nj = Math.min(this.params.anim['j blocks'], this.W)
    return {
      i: { n: ni, size: Math.ceil(this.H / ni), max: this.H },
      j: { n: nj, size: Math.ceil(this.W / nj), max: this.W },
    }
  }

  grid(dims, f) {
    grid(this.getBlockInfo(), dims, f)
  }

  getDispH() {
    const { i: { n, size } } = this.getBlockInfo()
    return this.H + Math.min(n, Math.ceil(this.H / size)) - 1
  }

  getDispW() {
    const { j: { n, size } } = this.getBlockInfo()
    return this.W + Math.min(n, Math.ceil(this.W / size)) - 1
  }

  initViz() {
    this.points = emptyPoints(this.H, this.W, this.getBlockInfo())
    this.points.name = `${this.params.name}.points`

    this.setColorsAndSizes()

    this.inner_group = new THREE.Group()
    this.inner_group.name = `${this.params.name}.inner_group`
    this.inner_group.add(this.points)

    const gap = this.params.layout.gap
    util.updateProps(this.inner_group.position, { x: gap, y: gap })

    this.group = new THREE.Group()
    this.group.name = `${this.params.name}.group`
    this.group.add(this.inner_group)

    this.setLegends()
  }

  setColorsAndSizes(r = undefined, c = undefined, size = undefined, color = undefined) {
    const [rstart, rend] = toRange(r, this.H)
    const [cstart, cend] = toRange(c, this.W)
    size = size || this.sizeFromData.bind(this)
    color = color || this.colorFromData.bind(this)
    for (let i = rstart; i < rend; i++) {
      for (let j = cstart; j < cend; j++) {
        const x = this.getData(i, j)
        this.setSize(i, j, size(x))
        this.setColor(i, j, color(x))
        this.checkLabel(i, j, x)
      }
    }
  }

  getExtent() {
    const gap = this.params.layout.gap
    return this._extents || (this._extents = {
      x: this.getDispW() + 2 * gap - 1,
      y: this.getDispH() + 2 * gap - 1,
      z: 0,
    })
  }

  getRangeInfo() {
    const viz = this.params.viz
    const use_absmin = viz.sensitivity == 'superlocal'
    const local_sens = use_absmin || viz.sensitivity == 'local'
    const absmax = local_sens ? this.absmax : this.getGlobalAbsmax()
    const absmin = use_absmin ? this.absmin : 0
    const absdiff = absmax - absmin
    if (absmin > absmax) {
      console.log(`HEY absmin ${absmin} > absmax ${absmax}`)
    }
    return { viz, absmin, absmax, absdiff }
  }

  sizeFromData(x) {
    if (x === undefined || isNaN(x)) {
      console.log(`HEY sizeFromData(${x})`)
      return 0
    }

    const absx = Math.abs(x)
    if (absx === Infinity) {
      return ELEM_SIZE
    }

    const { viz, absmin, absmax, absdiff } = this.getRangeInfo()
    const vol = absmax <= absmin ? 0 : (absx - absmin) / absdiff
    const zsize = viz['min size'] * ELEM_SIZE
    const size = zsize + (ELEM_SIZE - zsize) * Math.cbrt(vol)

    if (absx > absmax || size < 0 || size > ELEM_SIZE || isNaN(size)) {
      console.log(`HEY x ${x} size ${size} absx ${absx} absmax ${absmax} absmin ${absmin} zsize ${zsize}`)
    }

    return size
  }

  colorFromData(x) {
    if (x === undefined || isNaN(x)) {
      console.log(`HEY colorFromData(${x})`)
      return COLOR_TEMP.setHSL(0.0, 1.0, 1.0)
    }

    const absx = Math.abs(x)
    if (absx === Infinity) {
      return COLOR_TEMP.setHSL(1.0, 1.0, 1.0)
    }

    const { viz, absmin, absdiff } = this.getRangeInfo()

    const hue_vol = absdiff <= 0 ? 0 : (x - Math.sign(x) * absmin) / absdiff
    const gap = viz['hue gap'] * Math.sign(x)
    const hue = (viz['zero hue'] + gap + (Math.cbrt(hue_vol) * viz['hue spread'])) % 1

    const light_vol = absdiff <= 0 ? 0 : (absx - absmin) / absdiff
    const range = viz['max light'] - viz['min light']
    const light = viz['min light'] + range * Math.cbrt(light_vol)

    return COLOR_TEMP.setHSL(hue, 1.0, light)
  }

  getAbsmax() {
    return this.absmax
  }

  getGlobalAbsmax() {
    return this.params.getGlobalAbsmax ? this.params.getGlobalAbsmax() : this.absmax
  }

  reinit(init, epi = undefined, r = undefined, c = undefined) {
    this.data.reinit(init, epi, r, c)
    if (this.params.stretch_absmax) {
      this.absmax = Math.max(this.absmax, this.data.absmax())
      this.absmin = Math.min(this.absmin, this.data.absmin())
    }
    this.setColorsAndSizes(r, c)
  }

  getData(i, j) {
    if (i >= this.H || j >= this.W) {
      throw Error(`HEY i ${i} >= this.H ${this.H} || j ${j} >= this.W ${this.W}`)
    }
    return this.data.get(i, j)
  }

  getColor(i, j) {
    const colors = this.points.geometry.attributes.pointColor.array
    return COLOR_TEMP.fromArray(colors, this.data.addr(i, j) * 3)
  }

  setColor(i, j, c) {
    const colors = this.points.geometry.attributes.pointColor.array
    c.toArray(colors, this.data.addr(i, j) * 3)
    this.points.geometry.attributes.pointColor.needsUpdate = true
  }

  getSize(i, j) {
    return this.points.geometry.attributes.pointSize.array[this.data.addr(i, j)]
  }

  setSize(i, j, x) {
    this.points.geometry.attributes.pointSize.array[this.data.addr(i, j)] = x
    this.points.geometry.attributes.pointSize.needsUpdate = true
  }

  show(r = undefined, c = undefined) {
    this.setColorsAndSizes(r, c)
  }

  hide(r = undefined, c = undefined) {
    this.setColorsAndSizes(r, c, _ => 0, _ => ZERO_COLOR)
  }

  isHidden(i, j) {
    return this.getColor(i, j).equals(ZERO_COLOR)
  }

  bumpColor(r = undefined, c = undefined) {
    COLOR_TEMP.set(0x808080)
    this.setColorsAndSizes(r, c, undefined, x => this.colorFromData(x).add(COLOR_TEMP))
  }

  isFacing() {
    const c = this.group.localToWorld(new THREE.Vector3()).sub(this.context.camera.position).normalize()
    const m = this.group.getWorldDirection(new THREE.Vector3())
    return m.angleTo(c) < Math.PI / 2
  }

  isRightSideUp() {
    const q = new THREE.Quaternion()
    const p = new THREE.Vector3(0, -1, 0).applyQuaternion(this.group.getWorldQuaternion(q))
    const c = new THREE.Vector3(0, 1, 0).applyQuaternion(this.context.camera.quaternion)
    return p.angleTo(c) < Math.PI / 2
  }

  setRowGuides(light = undefined) {
    const prev = this.params.deco['row guides']
    light = util.syncProp(this.params.deco, 'row guides', light)
    if (this.row_guide_groups && prev == light) {
      return
    }
    if (this.row_guide_groups) {
      this.row_guide_groups.forEach(g => {
        this.inner_group.remove(g)
        util.disposeAndClear(g)
      })
    }
    this.row_guide_groups = []
    if (light > 0.0) {
      this.grid('ij', (
        { start: i, extent: ix, index: ii },
        { start: j, extent: jx, index: ji }
      ) => {
        const g = util.rowGuide(ix, jx, light)
        util.updateProps(g.position, { x: j + ji, y: i + ii })
        this.inner_group.add(g)
        this.row_guide_groups.push(g)
      })
    }
  }

  setFlowGuide(light) { }

  setName(name) {
    util.syncProp(this.params, 'name', name)
    this.setLegends()
  }

  setLegends(size = undefined) {
    const facing = this.isFacing()
    const rsu = this.isRightSideUp()
    const name = this.params.name
    if ((size === undefined || size == this.params.deco.legends) &&
      this.legendState &&
      this.legendState.facing == facing &&
      this.legendState.rsu == rsu &&
      this.legendState.name == name) {
      return
    }
    size = util.syncProp(this.params.deco, 'legends', size)
    this.legendState = { facing, rsu, name }
    if (this.name_text) {
      this.inner_group.remove(this.name_text)
      util.disposeAndClear(this.name_text)
    }
    if (size > 0 && name) {
      const color = 0xCCCCFF
      size = Math.cbrt(Math.max(5, this.getDispH()) * Math.max(this.getDispW(), 5)) * size / 20
      this.name_text = this.context.getText(name, color, size)
      this.name_text.name = `${name}.name`
      const { h, w } = util.gbbhwd(this.name_text.geometry)
      this.name_text.geometry.rotateZ(rsu ? Math.PI : 0)
      this.name_text.geometry.rotateY(facing ? Math.PI : 0)
      const xdir = facing == rsu ? 1 : -1
      const ydir = rsu ? 1 : 0
      const zdir = facing ? 1 : -1
      this.name_text.geometry.translate(
        util.center(this.getDispW() - 1, xdir * w),
        ydir * h + util.center(this.getDispH() - 1, h),
        -zdir // * h / 2
      )
      this.inner_group.add(this.name_text)
    }
  }

  checkLabel(i, j, x) {
    if (this.label_cache) {
      const addr = this.data.addr(i, j)
      const label = this.label_cache[addr]
      if (label != undefined && label.value != x) {
        util.disposeAndClear(label)
        this.label_cache[addr] = undefined
      }
    }
  }

  updateLabels(spotlight = undefined) {
    spotlight = util.syncProp(this.params.deco, 'spotlight', spotlight)
    if (spotlight == 0) {
      if (this.label_group) {
        this.inner_group.remove(this.label_group)
        util.disposeAndClear(this.label_group)
        this.label_group = undefined
      }
    } else {
      if (!this.label_group) {
        this.label_group = new THREE.Group()
        this.label_group.name = `${this.params.name}.label_group`
        this.inner_group.add(this.label_group)
        this.label_cache = []
      } else {
        util.disposeAndClear(this.label_group)
      }
      const { i: { size: si }, j: { size: sj } } = this.getBlockInfo()
      this.context.raycaster.params.Points.threshold = spotlight
      const intersects = this.context.raycaster.intersectObject(this.points)
      intersects.forEach(x => {
        const index = x.index
        const i = Math.floor(index / this.W)
        const j = index % this.W
        if (!this.isHidden(i, j)) {
          const x = this.getData(i, j)
          if (x != 0 && !isNaN(x) && isFinite(x)) {
            let label = this.label_cache[index]
            const facing = this.isFacing()
            const rsu = this.isRightSideUp()
            if (!label || label.facing != facing || label.rsu != rsu) {
              const fsiz = 0.16 - 0.008 * Math.log10(Math.floor(1 + Math.abs(x)))
              label = this.context.getText(`${x.toFixed(4)}`, 0xffffff, fsiz)
              label.name = `${this.params.name}.label[${i}, ${j}]`
              label.value = x
              label.facing = facing
              label.rsu = rsu
              const zdir = facing ? 1 : -1
              label.geometry.rotateX(zdir * Math.PI)
              label.geometry.rotateY(facing ? 0 : Math.PI)
              label.geometry.rotateZ(rsu ? 0 : Math.PI)
              const { h, w } = util.gbbhwd(label.geometry)
              const disp_i = i + Math.floor(i / si)
              const disp_j = j + Math.floor(j / sj)
              label.geometry.translate(
                util.center(disp_j * 2, (rsu ? zdir : -zdir) * w),
                h + util.center(disp_i * 2, h),
                -zdir * 0.25
              )
              this.label_cache[index] = label
            }
            this.label_group.add(label)
          }
        }
      })
    }
  }
}

//
// MatMul
//

export const POLARITIES = ['positive', 'negative', 'positive/negative', 'negative/positive']
export const LEFT_PLACEMENTS = ['left', 'right', 'left/right', 'right/left']
export const RIGHT_PLACEMENTS = ['top', 'bottom', 'top/bottom', 'bottom/top']
export const RESULT_PLACEMENTS = ['front', 'back', 'front/back', 'back/front']
export const SENSITIVITIES = ['global', 'local', 'superlocal']
export const ANIM_ALG = [
  'none', 'dotprod (row major)', 'dotprod (col major)', 'axpy', 'vmprod', 'mvprod', 'vvprod',
]
export const FUSE_MODE = ['none', 'sync', 'async']

export class MatMul {

  constructor(params, context, init_viz = true) {
    this.params = params
    this.context = context

    this.group = new THREE.Group()
    this.group.name = `${this.params.name}.group`

    const height = p => p.matmul ? height(p.left) : p.h
    const width = p => p.matmul ? width(p.right) : p.w

    this.H = height(params.left)
    this.D = width(params.left)
    this.W = width(params.right)

    if (this.D != height(params.right)) {
      console.log(`HEY left width ${this.D} != right height ${height(params.right)}`)
    }

    this.initLeft()
    this.initRight()
    this.initResult()

    if (init_viz) {
      this.initViz()
    }
  }

  getDispH() {
    const { i: { n, size } } = this.getBlockInfo()
    return this.H + Math.min(n, Math.ceil(this.H / size)) - 1
  }

  getDispD() {
    const { j: { n, size } } = this.getBlockInfo()
    return this.D + Math.min(n, Math.ceil(this.D / size)) - 1
  }

  getDispW() {
    const { k: { n, size } } = this.getBlockInfo()
    return this.W + Math.min(n, Math.ceil(this.W / size)) - 1
  }

  disposeAll() {
    util.disposeAndClear(this.group)
  }

  prepChildParams(base = undefined) {
    base ||= util.copyTree(this.params)
    return {
      ...base,
      ...(base != this.params ? {
        anim: { ...this.params.anim, ...base.anim || {} },
        deco: { ...this.params.deco, ...base.deco || {} },
        layout: { ...this.params.layout, ...base.layout || {} },
        viz: { ...this.params.viz, ...base.viz || {} },
      } : {}),
      getGlobalAbsmax: this.getGlobalAbsmax.bind(this),
    }
  }

  initLeft() {
    const left_params = this.prepChildParams(this.params.left, info)
    if (left_params.matmul) {
      left_params.anim['i blocks'] = this.params.anim['i blocks']
      left_params.anim['k blocks'] = this.params.anim['j blocks']
      this.left = new MatMul(left_params, this.context, false)
    } else {
      const { init, min, max, dropout } = left_params
      const f = getInitFunc(init, min, max, dropout)
      const data = Array2D.fromInit(this.H, this.D, f)
      this.left = new Mat(data, left_params, this.context, false)
    }
  }

  initRight() {
    const right_params = this.prepChildParams(this.params.right)
    if (right_params.matmul) {
      right_params.anim['i blocks'] = this.params.anim['j blocks']
      right_params.anim['k blocks'] = this.params.anim['k blocks']
      this.right = new MatMul(right_params, this.context, false)
    } else {
      right_params.anim['i blocks'] = right_params.anim['j blocks']
      right_params.anim['j blocks'] = right_params.anim['k blocks']
      const { init, min, max, dropout } = right_params
      const f = getInitFunc(init, min, max, dropout)
      const data = Array2D.fromInit(this.D, this.W, f)
      this.right = new Mat(data, right_params, this.context, false)
    }
  }

  initResult() {
    const result_init = (i, j) => this.dotprod_val(i, j)
    const data = Array2D.fromInit(this.H, this.W, result_init, this.params.epilog)
    // const params = {
    //   ...this.getLeafParams(),
    //   // TODO clean up
    //   // node_height: this.params.node_height,
    //   // node_depth: this.params.node_depth,
    //   // max_depth: this.params.max_depth,
    //   // count: this.params.count
    // }
    const result_params = this.prepChildParams()
    result_params.anim['i blocks'] = result_params.anim['i blocks']
    result_params.anim['j blocks'] = result_params.anim['k blocks']
    this.result = new Mat(data, result_params, this.context, false)
  }

  dotprod_val(i, k, minj = undefined, maxj = undefined) {
    if (minj === undefined) {
      minj = 0
    }
    if (maxj === undefined) {
      maxj = this.D
    }
    let x = 0.0
    for (let j = minj; j < maxj; j++) {
      const l = this.left.getData(i, j)
      const r = this.right.getData(j, k)
      x += l * r
    }
    if (isNaN(x)) {
      console.log(`HEY dotprod_val(${i}, ${k}, ${minj}, ${maxj}) is NaN`)
      return 0
    }
    const epi = this.params.epilog
    return epi == 'x/j' ? x / this.D :
      epi == 'x/sqrt(j)' || epi == 'softmax(x/sqrt(j))' ? x / Math.sqrt(this.D) :
        epi == 'tanh' ? Math.tanh(x) :
          epi == 'relu' ? Math.max(0, x) :
            x
  }

  getData(i, j) {
    return this.result.getData(i, j)
  }

  show(r = undefined, c = undefined) {
    this.left.show(r, c)
    this.right.show(r, c)
    this.result.show(r, c)
  }

  hide(r = undefined, c = undefined) {
    this.left.hide(r, c)
    this.right.hide(r, c)
    this.result.hide(r, c)
  }

  setColorsAndSizes(r = undefined, c = undefined, size = undefined, color = undefined) {
    this.result.setColorsAndSizes(r, c, size, color)
  }

  bumpColor(r = undefined, c = undefined) {
    this.result.bumpColor(r, c)
  }

  ijkmul(i, j, k) {
    return this.left.getData(i, j) * this.right.getData(j, k)
  }

  getExtent() {
    const gap = this.params.layout.gap
    return this._extents || (this._extents = {
      x: this.getDispW() + 2 * gap - 1,
      y: this.getDispH() + 2 * gap - 1,
      z: this.getDispD() + 2 * gap - 1,
    })
  }

  initViz(params = undefined) {
    if (params) {
      this.params = params
    }

    util.disposeAndClear(this.group)
    this.flow_guide_group = undefined
    this.anim_mats = []

    function first(x) {
      const sep = x.indexOf('/')
      return sep == -1 ? x : x.slice(0, sep)
    }

    function next(x) {
      const sep = x.indexOf('/')
      return sep == -1 ? x : x.slice(sep + 1) + '/' + x.slice(0, sep)
    }

    // ---

    if (this.left.params.anim.alg == 'none' || this.left.params.anim.alg == 'default') {
      this.left.params.anim.alg = this.params.anim.alg
    }
    if (this.right.params.anim.alg == 'none' || this.right.params.anim.alg == 'default') {
      this.right.params.anim.alg = this.params.anim.alg
    }

    // ---

    const is_left = this.params.is_left === undefined || this.params.is_left
    this.left.params.is_left = true
    this.right.params.is_left = false

    const negate = x => (
      x == 'negative' ? 'positive' : x == 'positive' ? 'negative' :
        x == 'left' ? 'right' : x == 'right' ? 'left' :
          x == 'top' ? 'bottom' : x == 'bottom' ? 'top' :
            x == 'front' ? 'back' : x == 'back' ? 'front' : undefined
    )

    // if (this.left.params.matmul) {
    //   console.log(`HEY L`)
    //   this.left.params.layout.polarity = negate(this.params.layout.polarity)

    //   // correct for .L+
    //   if (this.params.layout.polarity == 'negative' || is_left) {
    //     console.log(`HEY L 1a - ${this.params.layout.polarity == 'negative'} is_left ${is_left}`)
    //     this.left.params.layout['left placement'] = this.params.layout['left placement']
    //     this.left.params.layout['right placement'] = negate(this.params.layout['right placement'])
    //     this.left.params.layout['result placement'] = negate(this.params.layout['result placement'])
    //   } else {
    //     console.log(`HEY L 1b pol ${this.params.layout.polarity} is_left ${is_left}`)
    //     this.left.params.layout['left placement'] = negate(this.params.layout['left placement'])
    //     this.left.params.layout['right placement'] = negate(this.params.layout['right placement'])
    //     this.left.params.layout['result placement'] = this.params.layout['result placement']
    //   }
    // }

    // if (this.right.params.matmul) {
    //   console.log(`HEY R`)
    //   this.right.params.layout.polarity = negate(this.params.layout.polarity)

    //   // correct for .R+
    //   if (this.params.layout.polarity == 'negative') {
    //     console.log(`HEY R 1a - ${this.params.layout.polarity == 'negative'}`)
    //     this.right.params.layout['left placement'] = negate(this.params.layout['left placement'])
    //     this.right.params.layout['right placement'] = this.params.layout['right placement']
    //     this.right.params.layout['result placement'] = negate(this.params.layout['result placement'])
    //   } else { // +
    //     if (is_left) {
    //       console.log(`HEY R 1b pol ${this.params.layout.polarity} is_left ${is_left}`)
    //       this.right.params.layout['left placement'] = negate(this.params.layout['left placement'])
    //       this.right.params.layout['right placement'] = negate(this.params.layout['right placement'])
    //       this.right.params.layout['result placement'] = this.params.layout['result placement']
    //     }
    //   }
    // }

    const layout_desc = p => {
      const pol = { 'positive': '+', 'negative': '-', 'positive/negative': '+/-', 'negative/positive': '-/+' }[p.layout.polarity]
      const lfp = { 'left': 'L', 'right': 'R', 'left/right': 'L/R', 'right/left': 'R/L' }[p.layout['left placement']]
      const rtp = { 'top': 'T', 'bottom': 'B', 'top/bottom': 'T/B', 'bottom/top': 'B/T' }[p.layout['right placement']]
      const rsp = { 'front': 'F', 'back': 'B', 'front/back': 'F/B', 'back/front': 'B/F' }[p.layout['result placement']]
      return `${pol} ${lfp} ${rtp} ${rsp}`
    }

    // this.left.params.name += ` ${first(this.params.layout.polarity)}, ${first(this.params.layout['left placement'])}`
    // this.right.params.name += ` ${first(this.params.layout.polarity)}, ${first(this.params.layout['right placement'])}`
    // this.result.params.name += ` ${first(this.params.layout['result placement'])} (${layout_desc(this.params)})`

    // ---

    // result of a right of a left is back in current example
    // result of a right of a right is front

    // this.left.params.layout.polarity = next(this.params.layout.polarity)
    // this.left.params.layout['left placement'] = 'left/right'
    // this.left.params.layout['right placement'] = next(this.params.layout['right placement'])
    // if we're a right, this should be our result placement otherwise alt
    // this.left.params.layout['result placement'] = next(this.params.layout['result placement'])

    // this.right.params.layout.polarity = next(this.params.layout.polarity)
    // this.right.params.layout['left placement'] = next(this.params.layout['left placement'])
    // this.right.params.layout['right placement'] = 'top/bottom'
    // if we're a left, this should be our result placement otherwise alt
    // this.right.params.layout['result placement'] = next(this.params.layout['result placement'])

    // ---

    this.left.params.layout.polarity = next(this.params.layout.polarity)
    this.left.params.layout['left placement'] = next(this.params.layout['left placement'])
    this.left.params.layout['right placement'] = next(this.params.layout['right placement'])
    this.left.params.layout['result placement'] = next(this.params.layout['result placement'])

    this.right.params.layout.polarity = next(this.params.layout.polarity)
    this.right.params.layout['left placement'] = next(this.params.layout['left placement'])
    this.right.params.layout['right placement'] = next(this.params.layout['right placement'])
    this.right.params.layout['result placement'] = next(this.params.layout['result placement'])

    // ---

    this.initLeftVis()
    this.initRightVis()
    this.initResultVis()

    this.setFlowGuide()
    this.setRowGuides()
  }

  initLeftVis() {
    this.left.initViz()
    if (this.params.layout.polarity.startsWith('positive')) {
      this.left.group.rotation.y = -Math.PI / 2
      this.left.group.position.x = this.params.layout['left placement'].startsWith('left') ?
        -this.getLeftScatter() :
        this.getExtent().x + this.left.getExtent().z + this.getLeftScatter()
    } else { // negative
      this.left.group.rotation.y = Math.PI / 2
      this.left.group.position.z = this.getExtent().z
      this.left.group.position.x = this.params.layout['left placement'].startsWith('left') ?
        -(this.left.getExtent().z + this.getLeftScatter()) :
        this.getExtent().x + this.getLeftScatter()
    }
    this.group.add(this.left.group)
  }

  initRightVis() {
    this.right.initViz()
    if (this.params.layout.polarity.startsWith('positive')) {
      this.right.group.rotation.x = Math.PI / 2
      this.right.group.position.y = this.params.layout['right placement'].startsWith('top') ?
        -this.getRightScatter() :
        this.getExtent().y + this.right.getExtent().z + this.getRightScatter()
    } else { // negative
      this.right.group.rotation.x = -Math.PI / 2
      this.right.group.position.z = this.getExtent().z
      this.right.group.position.y =
        this.params.layout['right placement'].startsWith('top') ?
          -(this.right.getExtent().z + this.getRightScatter()) :
          this.getExtent().y + this.getRightScatter()
    }
    this.group.add(this.right.group)
  }

  initResultVis() {
    this.result.initViz()
    this.result.group.position.z =
      this.params.layout['result placement'].startsWith('back') ?
        this.getExtent().z :
        0
    this.group.add(this.result.group)
  }

  getPlacementInfo() {
    return {
      polarity: this.params.layout.polarity.startsWith('positive') ? 1 : -1,
      left: this.params.layout['left placement'].startsWith('left') ? 1 : -1,
      right: this.params.layout['right placement'].startsWith('top') ? 1 : -1,
      result: this.params.layout['result placement'].startsWith('front') ? 1 : -1,
      gap: this.params.layout.gap,
      left_scatter: this.getLeftScatter(),
      right_scatter: this.getRightScatter(),
    }
  }

  setFlowGuide(light = undefined) {
    if (light != this.params.deco['flow guides']) {
      light = util.syncProp(this.params.deco, 'flow guides', light)
      if (this.flow_guide_group) {
        this.group.remove(this.flow_guide_group)
        util.disposeAndClear(this.flow_guide_group)
        this.flow_guide_group = undefined
      }
      if (light > 0.0) {
        this.flow_guide_group = util.flowGuide(
          this.getDispH(), this.getDispD(), this.getDispW(), this.getPlacementInfo(), light
        )
        this.group.add(this.flow_guide_group)
      }
    }
    this.left.setFlowGuide(light)
    this.right.setFlowGuide(light)
  }

  scatterFromCount(count) {
    const blast = (count >= this.params.layout.molecule) ? count ** this.params.layout.blast : 0
    return this.params.layout.scatter * blast
  }

  getLeftScatter() {
    return this.scatterFromCount(this.left.params.count)
  }

  getRightScatter() {
    return this.scatterFromCount(this.right.params.count)
  }

  updateLabels(params = undefined) {
    if (params) {
      this.params.deco.spotlight = params.deco.spotlight
      this.params.deco['interior spotlight'] = params.deco['interior spotlight']
    }

    const spotlight = this.params.deco.spotlight
    this.left.updateLabels(this.left.params.matmul ? params : spotlight)
    this.right.updateLabels(this.right.params.matmul ? params : spotlight)
    this.result.updateLabels(spotlight)

    const interior_spotlight = this.params.deco['interior spotlight'] ? spotlight : 0
    this.anim_mats.map(m => m.updateLabels(interior_spotlight))
  }

  getBoundingBox() {
    return new THREE.Box3().setFromObject(this.group)
  }

  center() {
    const c = this.getBoundingBox().getCenter(new THREE.Vector3())
    util.updateProps(this.group.position, c.negate())
  }

  getAbsmax() {
    return Math.max(this.left.getAbsmax(), this.right.getAbsmax(), this.result.getAbsmax())
  }

  getGlobalAbsmax() {
    return this.params.getGlobalAbsmax ? this.params.getGlobalAbsmax() : this.getAbsmax()
  }

  hideInputs(hide) {
    util.syncProp(this.params.anim, 'hide inputs', hide)
    if (this.params.left.matmul) {
      this.left.hideInputs(hide)
    } else if (this.params.anim.alg != 'none') {
      hide ? this.left.hide() : this.left.show()
    }
    if (this.params.right.matmul) {
      this.right.hideInputs(hide)
    } else if (this.params.anim.alg != 'none') {
      hide ? this.right.hide() : this.right.show()
    }
  }

  setRowGuides(light) {
    light = util.syncProp(this.params.deco, 'row guides', light)
    this.left.setRowGuides(light)
    this.right.setRowGuides(light)
    this.result.setRowGuides(light)
    this.anim_mats.forEach(m => m.setRowGuides(light))
  }

  setName(name) {
    name = util.syncProp(this.params, 'name', name)
    this.result.setName(name)
  }

  setLegends(enabled = undefined) {
    util.syncProp(this.params.deco, 'legends', enabled)
    this.left.setLegends(enabled)
    this.right.setLegends(enabled)
    this.result.setLegends(enabled)
  }

  // animation

  initAnimation(cb = undefined) {
    if (this.params.anim.alg == 'none') {
      if (this.params.anim['hide inputs']) {
        !this.params.left.matmul && this.left.show()
        !this.params.right.matmul && this.right.show()
      }
      return
    }

    const bumps = {
      'dotprod (row major)': () => this.getVmprodBump(true),
      'dotprod (col major)': () => this.getMvprodBump(true),
      'axpy': () => this.getVvprodBump(true),
      'mvprod': () => this.getMvprodBump(false),
      'vmprod': () => this.getVmprodBump(false),
      'vvprod': () => this.getVvprodBump(false),
    }

    const nj = this.getBlockInfo().j.n
    const nlk = () => this.left.getBlockInfo().k.n
    const nri = () => this.right.getBlockInfo().i.n

    const { alg, fuse } = this.params.anim

    let left_done = true, right_done = true

    this.alg_join = () => {
      const lalg = !this.params.left.matmul || left_done ? 'none' :
        (fuse == 'async' || this.left.getIndex() == this.getIndex() ?
          this.left.alg_join() :
          'mixed')

      const ralg = !this.params.right.matmul || right_done ? 'none' :
        (fuse == 'async' || this.right.getIndex() == this.getIndex() ?
          this.right.alg_join() :
          'mixed')

      const or_none = (a, b) => a == b || a == 'none'

      return (alg == 'vmprod' && or_none(lalg, 'vmprod') && ralg == 'none') ? 'vmprod' :
        (alg == 'mvprod' && lalg == 'none' && or_none(ralg, 'mvprod')) ? 'mvprod' :
          (alg == 'vvprod' && or_none(lalg, 'mvprod') && or_none(ralg, 'vmprod')) ? 'vvprod' :
            (lalg == 'none' && ralg == 'none') ? alg :
              'mixed'
    }

    const can_fuse = () => fuse != 'none' && this.alg_join() != 'mixed'

    const start = () => {
      const result_bump = bumps[alg]()

      this.bump = () => {
        const go = left_done && right_done || can_fuse()
        left_done || this.left.bump()
        right_done || this.right.bump()
        go && result_bump()
      }

      if (this.params.left.matmul) {
        left_done = false
        this.left.initAnimation(() => left_done = true)
      }

      if (this.params.right.matmul) {
        right_done = false
        this.right.initAnimation(() => right_done = true)
      }

      if (this.params.anim['hide inputs']) {
        this.left.hide()
        this.right.hide()
      }
      this.result.hide()

      !cb && this.bump()
    }

    this.onAnimDone = () => {
      this.clearAnimMats()
      nj > 1 && this.result.show()
      cb ? cb() : start()
    }

    start()
  }

  getBlockInfo() {
    const ni = Math.min(this.params.anim['i blocks'], this.H)
    const nj = Math.min(this.params.anim['j blocks'], this.D)
    const nk = Math.min(this.params.anim['k blocks'], this.W)
    return {
      i: { n: ni, size: Math.ceil(this.H / ni), max: this.H },
      j: { n: nj, size: Math.ceil(this.D / nj), max: this.D },
      k: { n: nk, size: Math.ceil(this.W / nk), max: this.W },
    }
  }

  grid(dims, f) {
    grid(this.getBlockInfo(), dims, f)
  }

  getAnimIntermediateParams() {
    const params = this.prepChildParams()
    delete params.name
    params.viz.sensitivity = 'local'
    params.stretch_absmax = true
    params.anim['i blocks'] = 1
    params.anim['j blocks'] = 1
    params.anim['k blocks'] = 1
    return params
  }

  getAnimResultParams() {
    const params = this.prepChildParams()
    delete params.name
    params.viz.sensitivity = 'local'
    params.stretch_absmax = true
    params.anim['i blocks'] = params.anim['i blocks']
    params.anim['j blocks'] = params.anim['k blocks']
    return params
  }

  clearAnimMats() {
    this.anim_mats.forEach(m => {
      this.group.remove(m.group)
      util.disposeAndClear(m.group)
    })
    this.anim_mats = []
  }

  getAnimResultMats() {
    const { j: { n: nj, size: sj } } = this.getBlockInfo()
    if (nj == 1) {
      this.result.params.stretch_absmax = true
      return [this.result]
    }
    const { gap, polarity, result } = this.getPlacementInfo()
    const { z: extz } = this.getExtent()
    const results = []
    this.grid('j', ({ start: j, end: je, index: ji }) => {
      const result_init = (i, k) => this.dotprod_val(i, k, j, je)
      const data = Array2D.fromInit(this.H, this.W, result_init)
      const mat = new Mat(data, this.getAnimResultParams(), this.context, true)
      mat.group.position.z = polarity > 0 ?
        result > 0 ?
          ji == 0 ? this.result.group.position.z : gap + j + Math.floor((j - 1) / sj) :
          ji == nj - 1 ? this.result.group.position.z : gap + je + Math.floor(j / sj) :
        result > 0 ?
          ji == nj - 1 ? this.result.group.position.z : extz - je - Math.floor(je / sj) + 1 - gap :
          ji == 0 ? this.result.group.position.z : extz - j - Math.floor((je - 1) / sj) + 1 - gap
      mat.setRowGuides()
      mat.hide()
      results.push(mat)
      this.group.add(mat.group)
      this.anim_mats.push(mat)
    })
    return results
  }

  getVmprodBump(sweep) {
    const { gap, polarity } = this.getPlacementInfo()
    const results = this.getAnimResultMats()

    const vmps = {}
    this.grid('ijk', (
      { start: i, index: ii },
      { start: j, extent: jx, index: ji },
      { start: k, extent: kx, index: ki }
    ) => {
      const vmpinit = (jii, kii) => this.ijkmul(i, j + jii, k + kii)
      const data = Array2D.fromInit(jx, sweep ? 1 : kx, vmpinit)
      const vmp = new Mat(data, this.getAnimIntermediateParams(), this.context, true)
      vmp.hide()
      const z = polarity < 0 ? this.getExtent().z - j - ji : j + ji
      util.updateProps(vmp.group.position, { x: k + ki, y: gap + i + ii, z })
      vmp.group.rotation.x = polarity * Math.PI / 2
      vmps[[i, j, k]] = vmp
      this.anim_mats.push(vmp)
      this.group.add(vmp.group)
    })

    const { i: { size: isize }, k: { size: ksize } } = this.getBlockInfo()
    let curi = -1
    let curk = sweep ? -1 : 0

    this.getIndex = () => curi

    return () => {
      // update indexes
      const [oldi, oldk] = [curi, curk]
      sweep && (curk = (curk + 1) % ksize)
      curk == 0 && curi++

      // clear old input hilights
      if (oldi >= 0 && !this.params.anim['hide inputs']) {
        sweep && this.grid('k', ({ start: k, extent: kx }) => {
          oldk < kx && this.right.setColorsAndSizes(undefined, k + oldk)
        })
        oldi != curi && this.grid('i', ({ start: i, extent: ix }) => {
          oldi < ix && this.left.setColorsAndSizes(i + oldi, undefined)
        })
      }

      // end of cycle
      if (curi == isize && curk == sweep ? ksize : 0) {
        this.onAnimDone()
        return
      }

      // start of cycle
      if (curi == 0 && curk == 0) {
        Object.values(vmps).forEach(vmp => vmp.setRowGuides())
        results.forEach(r => r.hide())
      }

      // new input hilights
      if (!this.params.anim['hide inputs']) {
        sweep && this.grid('k', ({ start: k, extent: kx }) => {
          curk < kx && this.right.bumpColor(undefined, k + curk)
        })
        oldi != curi && this.grid('i', ({ start: i, extent: ix }) => {
          curi < ix && this.left.bumpColor(i + curi, undefined)
        })
      }

      // update intermediates
      this.grid('ijk', (
        { start: i, extent: ix, index: ii },
        { start: j },
        { start: k, extent: kx, index: ki }
      ) => {
        const vmp = vmps[[i, j, k]]
        if (curi < ix && curk < kx) {
          util.updateProps(vmp.group.position, { x: k + ki + curk, y: gap + i + ii + curi })
          vmp.reinit((ji, ki) => this.ijkmul(i + curi, j + ji, k + curk + ki))
        }
      })

      // reveal new results
      this.grid('ik', ({ start: i, extent: ix }, { start: k, end: ke, extent: kx }) => {
        curi < ix && curk < kx && results.forEach(r => r.show(i + curi, sweep ? k + curk : [k, ke]))
      })

      // update labels
      this.updateLabels()
    }
  }

  getMvprodBump(sweep) {
    const { gap, polarity } = this.getPlacementInfo()
    const results = this.getAnimResultMats()

    const mvps = {}
    this.grid('ijk', (
      { start: i, extent: ix, index: ii },
      { start: j, extent: jx, index: ji },
      { start: k }
    ) => {
      const mvpinit = (iii, jii) => this.ijkmul(i + iii, j + jii, k)
      const data = Array2D.fromInit(sweep ? 1 : ix, jx, mvpinit)
      const mvp = new Mat(data, this.getAnimIntermediateParams(), this.context, true)
      mvp.hide()
      const z = polarity < 0 ? this.getExtent().z - j - ji : j + ji
      util.updateProps(mvp.group.position, { x: gap + k + ji, y: i + ii, z })
      mvp.group.rotation.y = polarity * -Math.PI / 2
      mvps[[i, j, k]] = mvp
      this.anim_mats.push(mvp)
      this.group.add(mvp.group)
    })

    const { i: { size: isize }, k: { size: ksize } } = this.getBlockInfo()
    let curk = -1
    let curi = sweep ? -1 : 0

    this.getIndex = () => curk

    return () => {
      // update indexes
      const [oldi, oldk] = [curi, curk]
      sweep && (curi = (curi + 1) % isize)
      curi == 0 && curk++

      // clear old input hilights
      if (oldk >= 0 && !this.params.anim['hide inputs']) {
        sweep && this.grid('i', ({ start: i, extent: ix }) => {
          oldi < ix && this.left.setColorsAndSizes(i + oldi, undefined)
        })
        oldk != curk && this.grid('k', ({ start: k, extent: kx }) => {
          oldk < kx && this.right.setColorsAndSizes(undefined, k + oldk)
        })
      }

      // end of cycle
      if (curk == ksize && curi == sweep ? isize : 0) {
        this.onAnimDone()
        return
      }

      // start of cycle
      if (curk == 0 && curi == 0) {
        Object.values(mvps).forEach(vmp => vmp.setRowGuides())
        results.forEach(r => r.hide())
      }

      // new input hilights
      if (!this.params.anim['hide inputs']) {
        sweep && this.grid('i', ({ start: i, extent: ix }) => {
          curi < ix && this.left.bumpColor(i + curi, undefined)
        })
        oldk != curk && this.grid('k', ({ start: k, extent: kx }) => {
          curk < kx && this.right.bumpColor(undefined, k + curk)
        })
      }

      // update intermediates
      this.grid('ijk', (
        { start: i, extent: ix, index: ii },
        { start: j },
        { start: k, extent: kx, index: ki }
      ) => {
        const mvp = mvps[[i, j, k]]
        if (curi < ix && curk < kx) {
          util.updateProps(mvp.group.position, { x: gap + k + ki + curk, y: i + ii + curi })
          mvp.reinit((ii, ji) => this.ijkmul(i + curi + ii, j + ji, k + curk))
        }
      })

      // reveal new results
      this.grid('ik', ({ start: i, end: ie, extent: ix }, { start: k, extent: kx }) => {
        curi < ix && curk < kx && results.forEach(r => r.show(sweep ? i + curi : [i, ie], k + curk))
      })

      // update labels
      this.updateLabels()
    }
  }

  getVvprodBump(sweep) {
    const { gap, polarity } = this.getPlacementInfo()
    const { z: extz } = this.getExtent()
    const results = this.getAnimResultMats()

    const vvps = {}
    this.grid('ijk', (
      { start: i, extent: ix, index: ii },
      { start: j, index: ji },
      { start: k, extent: kx, index: ki }
    ) => {
      const vvpinit = (iii, kii) => this.ijkmul(i + iii, j, k + kii)
      const data = Array2D.fromInit(ix, sweep ? 1 : kx, vvpinit)
      const vvp = new Mat(data, this.getAnimIntermediateParams(), this.context, true)
      vvp.hide()
      const z = polarity > 0 ? gap + j + ji : extz - gap - j - ji
      util.updateProps(vvp.group.position, { x: k + ki, y: i + ii, z })
      vvps[[i, j, k]] = vvp
      this.anim_mats.push(vvp)
      this.group.add(vvp.group)
    })

    const { j: { size: jsize }, k: { size: ksize } } = this.getBlockInfo()
    let curj = -1
    let curk = sweep ? -1 : 0

    this.getIndex = () => curj

    return () => {
      // update indexes
      const [oldj, oldk] = [curj, curk]
      curj++
      if (sweep && curj % jsize == 0) {
        curj = 0
        curk++
      }

      // clear old input highlights
      if (oldj >= 0 && !this.params.anim['hide inputs']) {
        sweep ?
          this.grid('jk', ({ start: j, extent: jx }, { start: k, extent: kx }) => {
            oldj < jx && oldk < kx && this.right.setColorsAndSizes(j + oldj, k + oldk)
          }) :
          this.grid('j', ({ start: j, extent: jx }) => {
            oldj < jx && this.right.setColorsAndSizes(j + oldj, undefined)
          })
        this.grid('j', ({ start: j, extent: jx }) => {
          oldj < jx && this.left.setColorsAndSizes(undefined, j + oldj)
        })
      }

      // end of cycle
      if (sweep ? curk == ksize : curj == jsize) {
        this.onAnimDone()
        return
      }

      // start of cycle
      if (curk == 0 && curj == 0) {
        Object.values(vvps).forEach(vvp => vvp.setRowGuides())
        results.forEach(r => r.hide())
      }

      // new input highlights
      if (oldj >= 0 && !this.params.anim['hide inputs']) {
        sweep ?
          this.grid('jk', ({ start: j, extent: jx }, { start: k, extent: kx }) => {
            curj < jx && curk < kx && this.right.bumpColor(j + curj, k + curk)
          }) :
          this.grid('j', ({ start: j, extent: jx }) => {
            curj < jx && this.right.bumpColor(j + curj, undefined)
          })
        this.grid('j', ({ start: j, extent: jx }) => {
          curj < jx && this.left.bumpColor(undefined, j + curj)
        })
      }

      // update intermediates
      this.grid('ijk', (
        { start: i },
        { start: j, extent: jx, index: ji },
        { start: k, extent: kx, index: ki }
      ) => {
        const vvp = vvps[[i, j, k]]
        if (curj < jx && curk < kx) {
          const z = polarity > 0 ? gap + j + ji + curj : extz - gap - j - ji - curj
          vvp.group.position.z = z
          vvp.reinit((iii, kii) => this.ijkmul(i + iii, j + curj, k + curk + kii))
        }
      })

      // accumulate new results
      this.grid('jk', ({ start: j, extent: jx, index: ji }, { start: k, end: ke, extent: kx }) => {
        if (curj < jx && curk < kx) {
          const f = (ii, ki) => this.dotprod_val(ii, ki, j, j + curj + 1)
          results[ji].reinit(f, undefined, undefined, sweep ? k + curk : [k, ke])
        }
      })

      // update labels
      this.updateLabels()
    }
  }
}

export class Series {
  constructor(items, offset) {
    this.items = items
    this.group = new THREE.Group()
    const loc = new THREE.Vector3(0, 0, 0)
    this.items.forEach(item => {
      util.updateProps(item.group.position, loc)
      const { h, w, d } = util.bbhwd(item.getBoundingBox())
      offset.x != 0 && (loc.x += w + offset.x)
      offset.y != 0 && (loc.y += h + offset.y)
      offset.z != 0 && (loc.z += d + offset.z)
      this.group.add(item.group)
    })
  }

  getBoundingBox() {
    return new THREE.Box3().setFromObject(this.group)
  }

  center() {
    const c = this.getBoundingBox().getCenter(new THREE.Vector3())
    util.updateProps(this.group.position, c.negate())
  }

  setLegends(enabled = undefined) {
    this.items.forEach(item => item.setLegends(enabled))
  }

  updateLabels(params = undefined) {
    this.items.forEach(item => item.updateLabels(params))
  }

  initAnimation() {
    this.items.forEach(item => item.initAnimation())
  }

  disposeAll() {
    util.disposeAndClear(this.group)
  }
}

//
// eval
//

class Env {
  constructor() {
    this.lhs = undefined
  }
}

class Var {
  constructor(name, env) {
    this.name = name;
    this.env = env
  }
  valueOf() {
    console.log(`HEY ${this.name}.valueOf()`)
    this.env.lhs = this.env.lhs ? [this.env.lhs, this] : this
    // if (this.env.params) {
    //   this.env.params = {
    //     ...this.env.params,
    //     'right name': this.name, 'result name': `${this.env.params['left name']} * ${this.name}`
    //   }
    // } else {
    //   this.env.params = { ...this.env.getDefaultParams(), 'left name': this.name }
    // }
    return 0
  }
}

class Expr {
  constructor(input) {
    let expr = input.replaceAll('@', '*')
    let done = false
    let limit = 0
    this.env = new Env()
    while (!done && limit++ < 100) {
      try {
        done = true
        eval(expr)
      } catch (e) {
        if (e.message.indexOf('is not defined') == -1) {
          throw e
        }
        done = false
        const name = /^([\w\-]+)/.exec(e.message)[0]
        console.log(`HEY name ${name}`)
        expr = `const ${name} = new Var('${name}', this.env);${expr}`
        console.log(`HEY expr '${expr}'`)
      }
    }
    if (limit == 1000) {
      throw Error(`error evaluating input '${input}`)
    }
    this.expr = expr
    console.log(`HEY this.expr = ${this.expr} env = ${this.envenv}`)
  }
}

