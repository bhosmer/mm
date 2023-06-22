"use strict"

import * as THREE from 'three'
import * as util from './util.js'

//
// shader
//

const TEXTURE = new THREE.TextureLoader().load('./assets/ball.png')

export const MATERIAL = new THREE.ShaderMaterial({
  uniforms: {
    color: { value: new THREE.Color(0xffffff) },
    pointTexture: { value: TEXTURE },
    mag: { value: 1.0 },
  },

  vertexShader: `
  uniform float mag;
  attribute float pointSize;
  attribute vec4 pointColor;
  varying vec4 vColor;

  void main() {
    vColor = pointColor;
    vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
    gl_PointSize = mag * pointSize / -mvPosition.z;
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
  'col major': (i, j, h, w) => h * w > 1 ? (j * h + i) / (h * w - 1) : 0,
  'pt linear': (i, j, h, w) => (2 * Math.random() - 1) / Math.sqrt(w),
  uniform: () => Math.random(),
  gaussian: () => gaussianRandom(0.5, 0.5),
  // sphere: (i, j, h, w) => sampleSphere([h, w]),
  'tril mask': (i, j) => j <= i ? 1 : 0,
  'triu mask': (i, j) => j >= i ? 1 : 0,
  eye: (i, j) => +(i == j),
  diff: (i, j) => i == j ? 1 : i == j + 1 ? -1 : 0,
}

export const INITS = Object.keys(INIT_FUNCS).concat(['url', 'expr'])

const USE_RANGE = ['rows', 'cols', 'row major', 'col major', 'uniform', 'gaussian']
const USE_DROPOUT = USE_RANGE.concat(['pt linear'])

export const useRange = name => USE_RANGE.indexOf(name) >= 0
export const useDropout = name => USE_DROPOUT.indexOf(name) >= 0

const DATA_CACHE = {}

function tryLoadData(data_url) {
  if (DATA_CACHE[data_url]) {
    return DATA_CACHE[data_url]
  }
  try {
    console.log(`loading data from ${data_url}...`)
    const url = new URL(data_url)
    const req = new XMLHttpRequest()
    req.open("GET", url, false)
    req.send(null)
    DATA_CACHE[url] = req.responseText.split(/\r?\n|\r/).map(l => l.split(',').map(s => +s))
    console.log(`done loading data from ${data_url}`)
    return DATA_CACHE[url]
  } catch (e) {
    console.log(`error loading data from URL '${data_url}' message '${e.message}`)
  }
}

function tryURLInit(url) {
  const data = tryLoadData(url)
  if (data) {
    return (i, j, h, w) => {
      const row = data[i % data.length]
      return row[j % row.length]
    }
  }
}

function tryEvalInitExpr(expr) {
  try {
    return eval?.(`(i, j, h, w) => (${expr})`)
  } catch (e) {
    console.log(`error evaluating init expr '${expr}' message '${e.message}`)
  }
}

function getInitFunc(init_params) {
  const { init, min, max, dropout, url, expr } = init_params
  const f = INIT_FUNCS[init] ||
    (init == 'url' && tryURLInit(url)) ||
    (init == 'expr' && tryEvalInitExpr(expr))
  if (!f) {
    console.log(init == 'url' ?
      `'can't load from URL '${url}'` :
      `unrecognized initializer '${init}'`)
    return () => 0
  }
  const scaled = useRange(init) && (min != 0 || max != 1) ?
    (i, j, h, w) => min + Math.max(0, max - min) * f(i, j, h, w) :
    f
  const sparse = useDropout(init) && dropout > 0 ?
    (i, j, h, w) => Math.random() > dropout ? scaled(i, j, h, w) : 0 :
    scaled
  return sparse
}

// pointwise funcs

const ERF_A1 = 0.254829592
const ERF_A2 = -0.284496736
const ERF_A3 = 1.421413741
const ERF_A4 = -1.453152027
const ERF_A5 = 1.061405429
const ERF_P = 0.3275911

function erf(x) {
  const absx = Math.abs(x)
  const t = 1.0 / (1.0 + ERF_P * absx)
  const y = (((((ERF_A5 * t + ERF_A4) * t) + ERF_A3) * t + ERF_A2) * t + ERF_A1) * t
  return Math.sign(x) * (1 - y * Math.exp(-absx * absx))
}

const SQRT2 = Math.sqrt(2)

const gelu = x => x * (1 + erf(x / SQRT2)) / 2

const sigmoid = x => 1 / (1 + Math.exp(-x))

const silu = x => x * sigmoid(x)

const relu = x => Math.max(0, x)

const POINTWISE = {
  'relu': relu,
  'gelu': gelu,
  'sigmoid': sigmoid,
  'silu': silu,
  'tanh': Math.tanh,
}

// epilogs
// todo the way epis are done is kind of messy rn

export const EPILOGS = [
  'none',
  'relu',
  'gelu',
  'sigmoid',
  'silu',
  'tanh',
  'layernorm',
  'softmax',
  'softmax(x/sqrt(k))',
  'softmax(tril(x/sqrt(k)))',
  'x/k',
  'x/sqrt(k)',
]

function softmax_(h, w, data, tril = false) {

  const row_max = (ptr, w) => {
    let x = 0
    for (let j = 0; j < w; j++, ptr++) {
      x = Math.max(x, data[ptr])
    }
    return x
  }

  const calc_denom = (ptr, w, rmax) => {
    let d = 0
    for (let j = 0; j < w; j++, ptr++) {
      d += Math.exp(data[ptr] - rmax)
      if (!isFinite(d)) {
        // console.log(`HEY denom at data[${ptr}) = ${data[ptr]} becomes infinite`)
        break
      }
    }
    return d
  }

  for (let i = 0, ptr = 0; i < h; i++) {
    const rmax = row_max(ptr, tril ? i + 1 : w)
    const denom = calc_denom(ptr, tril ? i + 1 : w, rmax)
    for (let j = 0; j < w; j++, ptr++) {
      const x = tril && j > i ? 0 : Math.exp(data[ptr] - rmax) / denom
      if (isNaN(x)) {
        // console.log(`HEY Math.exp(data[${ptr}) = ${data[ptr]}]) / ${denom} is NaN`)
        data[ptr] = 0
      } else {
        data[ptr] = x
      }
    }
  }
}

const softmax_tril_ = (h, w, data) => softmax_(h, w, data, true)

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

const IN_PLACE_EPILOGS = {
  'softmax': softmax_,
  'softmax(x/sqrt(k))': softmax_,
  'softmax(tril(x/sqrt(k)))': softmax_tril_,
  'layernorm': layernorm_,
}

const getInPlaceEpilog = name => IN_PLACE_EPILOGS[name]

function applyInPlaceEpilog_(data, h, w, epi) {
  const epi_ = epi && getInPlaceEpilog(epi)
  if (epi_) {
    epi_(h, w, data)
  }
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
  applyInPlaceEpilog_(data, h, w, epi)
}

export class Array2D {

  static fromInit(h, w, init, epi = undefined) {
    const data = new Float32Array(h * w)
    initArrayData_(data, h, w, init, epi)
    return new Array2D(h, w, data)
  }

  constructor(h, w, data) {
    this.h = h | 0
    this.w = w | 0
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

let elem_scale = 1.25
let elem_size = elem_scale

function setElemScale(s) {
  s ||= elem_scale
  const old_elem_scale = elem_scale
  elem_scale = s
  elem_size *= elem_scale / old_elem_scale
}

export function setElemSize(scale, pixel_ratio) {
  elem_size = elem_scale * Math.min(scale.x, scale.y) * pixel_ratio
}

const ZERO_COLOR = new THREE.Color(0, 0, 0)
const COLOR_TEMP = new THREE.Color()

function emptyPoints(h, w, info) {
  const { i: { size: si }, j: { size: sj }, gap } = info
  const n = h * w
  const points = new Float32Array(n * 3)
  for (let i = 0, ptr = 0; i < h; i++) {
    const ioff = Math.floor(i / si)
    for (let j = 0; j < w; j++) {
      const joff = Math.floor(j / sj)
      points[ptr++] = j + joff * gap
      points[ptr++] = i + ioff * gap
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
    const ni = Math.min(this.params.block['i blocks'], this.H)
    const nj = Math.min(this.params.block['j blocks'], this.W)
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
    return this.H + this.params.layout.gap * (Math.min(n, Math.ceil(this.H / size)) - 1)
  }

  getDispW() {
    const { j: { n, size } } = this.getBlockInfo()
    return this.W + this.params.layout.gap * (Math.min(n, Math.ceil(this.W / size)) - 1)
  }

  initViz() {
    const gap = this.params.layout.gap
    const info = { ...this.getBlockInfo(), gap }

    this.points = emptyPoints(this.H, this.W, info)
    this.points.name = `${this.params.name}.points`

    this.setColorsAndSizes()

    this.inner_group = new THREE.Group()
    this.inner_group.name = `${this.params.name}.inner_group`
    this.inner_group.add(this.points)

    util.updateProps(this.inner_group.position, { x: gap, y: gap })

    this.group = new THREE.Group()
    this.group.name = `${this.params.name}.group`
    this.group.add(this.inner_group)

    this.setLegends()
  }

  setColorsAndSizes(r = undefined, c = undefined, get_size = undefined, get_color = undefined) {
    const [rstart, rend] = toRange(r, this.H)
    const [cstart, cend] = toRange(c, this.W)
    get_size = get_size || this.sizeFromData.bind(this)
    get_color = get_color || this.colorFromData.bind(this)
    for (let i = rstart; i < rend; i++) {
      for (let j = cstart; j < cend; j++) {
        const x = this.getData(i, j)
        this.setSize(i, j, get_size(x))
        this.setColor(i, j, get_color(x))
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

    const local_absmax = this.absmax
    const global_absmax = this.getGlobalAbsmax()
    const absmax = (use_absmin || viz.sensitivity == 'local') ? local_absmax :
      viz.sensitivity == 'global' ? global_absmax :
        Math.sqrt(local_absmax * global_absmax) // semilocal
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

    if (x === 0) {
      return 0
    }

    const absx = Math.abs(x)
    if (absx === Infinity) {
      return elem_size
    }

    const { viz, absmin, absmax, absdiff } = this.getRangeInfo()
    const vol = absmax <= absmin ? 0 : (absx - absmin) / absdiff
    const zsize = viz['min size'] * elem_size
    const size = zsize + (elem_size - zsize) * Math.sqrt(vol)

    if (isNaN(size)) {
      this.n_size_from_data_errors = (this.n_size_from_data_errors || 0) + 1
      if (this.n_size_from_data_errors <= 100) {
        console.log(`HEY x ${x} size ${size} absx ${absx} absmax ${absmax} absmin ${absmin} zsize ${zsize}`)
        if (this.n_size_from_data_errors == 100) {
          console.log(`HEY stopping logging after 100 errors`)
        }
      }
    }

    // boundary violations can happen in intermediates
    return Math.min(size, elem_size)
  }

  colorFromData(x) {
    if (x === undefined || isNaN(x)) {
      console.log(`HEY colorFromData(${x})`)
      return COLOR_TEMP.setHSL(0.0, 1.0, 1.0)
    }

    if (x === 0) {
      return COLOR_TEMP.setHSL(0.0, 1.0, 0.0)
    }

    const { viz, absmin, absmax, absdiff } = this.getRangeInfo()

    // boundary violations can happen in intermediates
    const absx = Math.min(absmax, Math.max(absmin, Math.abs(x)))

    if (absx === Infinity) {
      return COLOR_TEMP.setHSL(1.0, 1.0, 1.0)
    }

    const hue_vol = absdiff <= 0 ? 0 : (x - Math.sign(x) * absmin) / absdiff
    const gap = viz['hue gap'] * Math.sign(x)
    const hue = (viz['zero hue'] + gap + (hue_vol * viz['hue spread'])) % 1

    const min_light = Math.max(viz['min light'], 0.00001)
    const max_light = Math.max(viz['max light'], min_light)
    const range = max_light - min_light
    const light_vol = absdiff <= 0 ? 0 : (absx - absmin)
    const light = min_light + range * Math.sqrt(light_vol) / Math.sqrt(absdiff)

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
    this.setColorsAndSizes(r, c)
  }

  getDataArray() {
    return this.data.data
  }

  getData(i, j) {
    if (i >= this.H || j >= this.W) {
      console.log(`HEY i ${i} >= this.H ${this.H} || j ${j} >= this.W ${this.W}`)
      return 0
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
      const gap = this.params.layout.gap
      this.grid('ij', (
        { start: i, extent: ix, index: ii },
        { start: j, extent: jx, index: ji }
      ) => {
        const g = util.rowGuide(ix, jx, light)
        util.updateProps(g.position, { x: j + ji * gap, y: i + ii * gap })
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

  setLegends(size = undefined, shape = undefined) {
    shape = util.syncProp(this.params.deco, 'shape', shape)
    const facing = this.isFacing()
    const rsu = this.isRightSideUp()
    const [H, W] = [this.H, this.W]
    const name = this.params.name // && this.params.name + (shape ? ` [${H}, ${W}]` : '')

    if ((size === undefined || size == this.params.deco.legends) &&
      this.legend_state &&
      this.legend_state.facing == facing &&
      this.legend_state.rsu == rsu &&
      this.legend_state.name == name &&
      this.legend_state.shape == shape &&
      this.legend_state.H == H && this.legend_state.W == W) {
      return
    }

    size = util.syncProp(this.params.deco, 'legends', size)
    this.legend_state = { facing, rsu, name, shape, H, W }
    const rmv = x => {
      if (x) {
        this.inner_group.remove(x)
        util.disposeAndClear(x)
      }
    }
    rmv(this.name_text)
    rmv(this.hdim_text)
    rmv(this.wdim_text)

    if (size > 0) {
      const color = 0xCCCCFF
      const adjsiz = size * Math.cbrt(H * W) / 10
      const xdir = facing ? 1 : -1
      const ydir = rsu ? 1 : 0
      const zdir = facing ? 1 : -1
      if (name) {
        const adjsiz2 = adjsiz * Math.min(1, 8 / name.length)
        this.name_text = util.getText(name, color, adjsiz2)
        this.name_text.name = `${name}.name`
        this.name_text.geometry.rotateZ(Math.PI)
        this.name_text.geometry.rotateY(facing ? Math.PI : 0)
        const { h, w } = util.gbbhwd(this.name_text.geometry)
        this.name_text.geometry.translate(
          util.center(this.getDispW() - 1, xdir * w),
          h + util.center(this.getDispH() - 1, h),
          -zdir
        )
        this.inner_group.add(this.name_text)
      }
      if (shape && this.params.deco.shape_info) {
        const htext = util.getText("X", color, adjsiz / 2.5)
        const { h } = util.gbbhwd(htext.geometry)
        util.disposeAndClear(htext)
        const { i: { n: ni }, j: { n: nj } } = this.getBlockInfo()
        {
          const { h: { name, place } } = this.params.deco.shape_info
          const hdim_str = `${name} = ${H}` + (ni == 1 ? '' : ` / ${ni}`)
          this.hdim_text = util.getText(hdim_str, color, adjsiz / 2.5)
          const { w } = util.gbbhwd(this.hdim_text.geometry)
          this.hdim_text.geometry.rotateZ((place == facing ? 1 : -1) * Math.PI / 2)
          this.hdim_text.geometry.rotateY(facing ? Math.PI : 0)
          const xgap = 2 * h
          this.hdim_text.geometry.translate(
            place ? this.getDispW() - 1 + xgap : -xgap,
            (place == facing ? 0 : w) + util.center(this.getDispH() - 1, w),
            0
          )
          this.inner_group.add(this.hdim_text)
        }
        {
          const { w: { name, place } } = this.params.deco.shape_info
          const wdim_str = `${name} = ${W}` + (nj == 1 ? '' : ` / ${nj}`)
          this.wdim_text = util.getText(wdim_str, color, adjsiz / 2.5)
          const { w } = util.gbbhwd(this.wdim_text.geometry)
          this.wdim_text.name = `${name}.wdim`
          this.wdim_text.geometry.rotateZ(Math.PI)
          this.wdim_text.geometry.rotateY(facing ? Math.PI : 0)
          this.wdim_text.geometry.translate(
            util.center(this.getDispW() - 1, (facing ? 1 : -1) * w),
            place ? this.getDispH() - 1 + 3 * h : -2 * h,
            0
          )
          this.inner_group.add(this.wdim_text)
        }
      }
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
      const gap = this.params.layout.gap
      const { i: { size: si }, j: { size: sj } } = this.getBlockInfo()
      this.context.raycaster.params.Points.threshold = spotlight
      const intersects = this.context.raycaster.intersectObject(this.points)
      let count = 0
      intersects.forEach(p => {
        const index = p.index
        const i = Math.floor(index / this.W)
        const j = index % this.W
        if (!this.isHidden(i, j)) {
          const x = this.getData(i, j)
          let label = this.label_cache[index]
          const facing = this.isFacing()
          const rsu = this.isRightSideUp()
          if (!label || label.facing != facing || label.rsu != rsu) {
            const fsiz = isNaN(x) || !isFinite(x) ? 0.12 :
              0.16 - 0.008 * Math.log10(Math.floor(1 + Math.abs(x)))
            label = util.getText(x.toFixed(5), 0xffffff, fsiz)
            count += 1
            // label.name = `${this.params.name}.label[${i}, ${j}]`
            label.value = x
            label.facing = facing
            label.rsu = rsu
            const zdir = facing ? 1 : -1
            label.geometry.rotateX(zdir * Math.PI)
            label.geometry.rotateY(facing ? 0 : Math.PI)
            label.geometry.rotateZ(rsu ? 0 : Math.PI)
            const { h, w } = util.gbbhwd(label.geometry)
            const disp_i = i + Math.floor(i / si) * gap
            const disp_j = j + Math.floor(j / sj) * gap
            label.geometry.translate(
              util.center(disp_j * 2, (rsu ? zdir : -zdir) * w),
              h + util.center(disp_i * 2, h),
              -zdir * 0.5
            )
            this.label_cache[index] = label
          }
          this.label_group.add(label)
        }
      })
    }
  }
}

//
// MatMul
//

export const SCHEMES = ['blocks', 'zigzag', 'wheel', 'custom']
export const POLARITIES = ['negative', 'positive']
export const LEFT_PLACEMENTS = ['left', 'right']
export const RIGHT_PLACEMENTS = ['top', 'bottom']
export const RESULT_PLACEMENTS = ['front', 'back']

function layoutDesc(layout) {
  const pol = { 'positive': '+', 'negative': '-', }[layout.polarity]
  const lfp = { 'left': 'L', 'right': 'R', }[layout['left placement']]
  const rtp = { 'top': 'T', 'bottom': 'B', }[layout['right placement']]
  const rsp = { 'front': 'F', 'back': 'B', }[layout['result placement']]
  return `${pol}${lfp}${rtp}${rsp}`
}

export const SENSITIVITIES = ['global', 'semilocal', 'local', 'superlocal']
export const TOP_LEVEL_ANIM_ALGS = [
  'none', 'dotprod (row major)', 'dotprod (col major)', 'axpy', 'vmprod', 'mvprod', 'vvprod',
]
export const ANIM_ALGS = TOP_LEVEL_ANIM_ALGS.concat('inherit')
export const FUSE_MODE = ['none', 'sync', 'async']

const ensureChildCounts = p => {
  if (p.count === undefined) {
    p.count = p.matmul === false ? 0 :
      (1 + ensureChildCounts(p.left).count + ensureChildCounts(p.right).count)
    // sloppy - this means root
    if (p.matmul === undefined) {
      const total = p.count
      const setTotal = p => {
        p.total = total
        p.left && setTotal(p.left)
        p.right && setTotal(p.right)
      }
      setTotal(p)
    }
  }
  return p
}

export class MatMul {

  constructor(params, context, init_viz = true) {
    this.context = context

    this.params = util.copyTree(params)
    ensureChildCounts(this.params)

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
    return this.H + this.params.layout.gap * (Math.min(n, Math.ceil(this.H / size)) - 1)
  }

  getDispD() {
    const { k: { n, size } } = this.getBlockInfo()
    return this.D + this.params.layout.gap * (Math.min(n, Math.ceil(this.D / size)) - 1)
  }

  getDispW() {
    const { j: { n, size } } = this.getBlockInfo()
    return this.W + this.params.layout.gap * (Math.min(n, Math.ceil(this.W / size)) - 1)
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
        block: { ...this.params.block, ...base.block || {} },
        deco: { ...this.params.deco, ...base.deco || {} },
        layout: { ...this.params.layout, ...base.layout || {} },
        viz: { ...this.params.viz, ...base.viz || {} },
      } : {}),
      getGlobalAbsmax: this.getGlobalAbsmax.bind(this),
    }
  }

  initLeft() {
    const left_params = this.prepChildParams(this.params.left)
    left_params.is_child = 'left'
    left_params.block['i blocks'] = this.params.block['i blocks']
    left_params.block['j blocks'] = this.params.block['k blocks']
    if (left_params.matmul) {
      this.left = new MatMul(left_params, this.context, false)
    } else {
      const { right, result, polarity } = this.getPlacementInfo()
      left_params.deco.shape_info = {
        h: { name: 'I', place: result == polarity },
        w: { name: 'K', place: right },
      }
      const data = Array2D.fromInit(this.H, this.D, getInitFunc(left_params))
      this.left = new Mat(data, left_params, this.context, false)
    }
  }

  initRight() {
    const right_params = this.prepChildParams(this.params.right)
    right_params.is_child = 'right'
    right_params.block['i blocks'] = this.params.block['k blocks']
    right_params.block['j blocks'] = this.params.block['j blocks']
    if (right_params.matmul) {
      this.right = new MatMul(right_params, this.context, false)
    } else {
      const { left, result, polarity } = this.getPlacementInfo()
      right_params.deco.shape_info = {
        h: { name: 'K', place: left },
        w: { name: 'J', place: result == polarity },
      }
      const data = Array2D.fromInit(this.D, this.W, getInitFunc(right_params))
      this.right = new Mat(data, right_params, this.context, false)
    }
  }

  initResult() {
    const result_init = (i, j) => this.dotprod(i, j, 0, this.D)
    const data = Array2D.fromInit(this.H, this.W, result_init, this.params.epilog)
    const result_params = this.prepChildParams()
    // if (this.params.total == this.params.count) {
    if (!this.params.is_child) {
      const placement = this.getPlacementInfo()
      result_params.deco.shape_info = {
        h: { name: 'I', place: placement.left },
        w: { name: 'J', place: placement.right },
      }
    }
    result_params.block['i blocks'] = result_params.block['i blocks']
    result_params.block['j blocks'] = result_params.block['j blocks']
    this.result = new Mat(data, result_params, this.context, false)
  }

  // todo clean up the way epilogs are done
  applyPointwiseEpilog(x) {
    const epi = this.params.epilog
    const pw = POINTWISE[epi]
    return epi == 'x/k' ? x / this.D :
      epi.includes('x/sqrt(k)') ? x / Math.sqrt(this.D) :
        pw ? pw(x) : x
  }

  dotprod(i, k, minj, maxj) {
    const lw = this.left.W
    const ld = this.left.getDataArray()
    const rw = this.right.W
    const rd = this.right.getDataArray()
    const maxlx = i * lw + maxj

    let x = 0.0
    for (let lx = i * lw + minj, rx = minj * rw + k; lx < maxlx; lx++, rx += rw) {
      x += ld[lx] * rd[rx]
    }

    if (isNaN(x)) {
      console.log(`HEY dotprod(${i}, ${k}, ${minj}, ${maxj}) is NaN`)
      return 0
    }

    return this.applyPointwiseEpilog(x, this.params.epilog)
  }

  getDataArray() {
    return this.result.getDataArray()
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

  ikjmul(i, k, j) {
    return this.left.getData(i, k) * this.right.getData(k, j)
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

    if (this.left.params.anim.alg == 'inherit') {
      this.left.params.anim.alg = this.params.anim.alg
    }
    if (this.right.params.anim.alg == 'inherit') {
      this.right.params.anim.alg = this.params.anim.alg
    }

    setElemScale(this.params.viz['elem scale'])
    this.initResultViz()
    this.initLeftViz()
    this.initRightViz()

    this.setFlowGuide()
    this.setRowGuides()
  }

  initLeftViz() {
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

  initRightViz() {
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

  initResultViz() {
    this.result.initViz()
    this.result.group.position.z =
      this.params.layout['result placement'].startsWith('back') ?
        this.getExtent().z :
        0
    this.group.add(this.result.group)
  }

  getPlacementInfo() {
    return {
      polarity: this.params.layout.polarity.startsWith('positive'),
      left: this.params.layout['left placement'].startsWith('left'),
      right: this.params.layout['right placement'].startsWith('top'),
      result: this.params.layout['result placement'].startsWith('front'),
    }
  }

  getLayoutInfo() {
    const info = this.getPlacementInfo()
    Object.entries(info).forEach(([k, v]) => info[k] = v ? 1 : -1)
    info.gap = this.params.layout.gap
    info.left_scatter = this.getLeftScatter()
    info.right_scatter = this.getRightScatter()
    return info
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
          this.getDispH(), this.getDispD(), this.getDispW(), this.getLayoutInfo(), light
        )
        this.group.add(this.flow_guide_group)
      }
    }
    this.left.setFlowGuide(light)
    this.right.setFlowGuide(light)
  }

  scatterFromCount(count) {
    const { scatter, molecule, blast } = this.params.layout
    const mult = count < molecule ? 0 :
      blast >= 0 ? count ** blast :
        (this.params.total - count) ** -blast
    return scatter * mult
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
    const get_bb = mm => {
      const min = mm.group.localToWorld(new THREE.Vector3())
      const max = mm.group.localToWorld(new THREE.Vector3().copy(mm.getExtent()))
      const swap = d => { const temp = min[d]; min[d] = max[d]; max[d] = temp }
      ['x', 'y', 'z'].forEach(d => { if (min[d] > max[d]) swap(d) })
      let bb = new THREE.Box3(min, max)
      mm.params.left.matmul && bb.union(get_bb(mm.left))
      mm.params.right.matmul && bb.union(get_bb(mm.right))
      return bb
    }
    return get_bb(this)
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

  setLegends(name = undefined, shape = undefined) {
    name = util.syncProp(this.params.deco, 'legends', name)
    shape = util.syncProp(this.params.deco, 'shape', shape)
    this.left.setLegends(name, shape)
    this.right.setLegends(name, shape)
    this.result.setLegends(name, shape)
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

      if (this.params.left.matmul && this.params.left.anim.alg != 'none') {
        left_done = false
        this.left.initAnimation(() => left_done = true)
      }

      if (this.params.right.matmul && this.params.right.anim.alg != 'none') {
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
    const ni = Math.min(this.params.block['i blocks'], this.H)
    const nk = Math.min(this.params.block['k blocks'], this.D)
    const nj = Math.min(this.params.block['j blocks'], this.W)
    return {
      i: { n: ni, size: Math.ceil(this.H / ni), max: this.H },
      k: { n: nk, size: Math.ceil(this.D / nk), max: this.D },
      j: { n: nj, size: Math.ceil(this.W / nj), max: this.W },
    }
  }

  grid(dims, f) {
    grid(this.getBlockInfo(), dims, f)
  }

  getAnimIntermediateParams(name) {
    const params = this.prepChildParams()
    // params.name = name // debug
    delete params.name
    params.viz.sensitivity == 'superlocal' && (params.viz.sensitivity = 'local')
    params.block['i blocks'] = 1
    params.block['k blocks'] = 1
    params.block['j blocks'] = 1
    return params
  }

  getAnimResultParams() {
    const params = this.prepChildParams()
    // params.name = name // debug
    delete params.name
    params.viz.sensitivity == 'superlocal' && (params.viz.sensitivity = 'local')
    params.block['i blocks'] = params.block['i blocks']
    params.block['k blocks'] = params.block['j blocks']
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
    const { k: { n: nk, size: sk } } = this.getBlockInfo()
    if (nk == 1) {
      return [this.result]
    }
    const { gap, polarity, result } = this.getLayoutInfo()
    const { z: extz } = this.getExtent()
    const results = []
    this.grid('k', ({ start: k, end: ke, index: ki }) => {
      const result_init = (i, j) => this.dotprod(i, j, k, ke)
      const data = Array2D.fromInit(this.H, this.W, result_init)
      const mat = new Mat(data, this.getAnimResultParams(), this.context, true)
      mat.group.position.z = polarity > 0 ?
        result > 0 ?
          ki == 0 ?
            this.result.group.position.z :
            gap + k + Math.floor(gap * k / sk - gap / 4) :
          ki == nk - 1 ?
            this.result.group.position.z :
            gap + ke + Math.floor(gap * k / sk + (gap - 1) / 4) :
        result > 0 ?
          ki == nk - 1 ?
            this.result.group.position.z :
            extz - ke - Math.floor(gap * ke / sk + (gap - 1) / 4) :
          ki == 0 ?
            this.result.group.position.z :
            extz - k - Math.floor(gap * ke / sk - gap / 4)
      mat.setRowGuides()
      mat.hide()
      results.push(mat)
      this.group.add(mat.group)
      this.anim_mats.push(mat)
    })
    return results
  }

  getVmprodBump(sweep) {
    const { gap, polarity } = this.getLayoutInfo()
    const results = this.getAnimResultMats()

    const vmps = {}
    this.grid('ikj', (
      { start: i, index: ii },
      { start: k, extent: kx, index: ki },
      { start: j, extent: jx, index: ji }
    ) => {
      const vmpinit = (kii, jii) => this.ikjmul(i, k + kii, j + jii)
      const data = Array2D.fromInit(kx, sweep ? 1 : jx, vmpinit)
      const vmp = new Mat(data, this.getAnimIntermediateParams(this.params.name + `.vmp[${ii}, ${ki}, ${ji}]`), this.context, true)
      vmp.hide()
      const z = polarity < 0 ? this.getExtent().z - k - (gap * ki) : k + (gap * ki)
      util.updateProps(vmp.group.position, { x: j + ji, y: gap + i + ii, z })
      vmp.group.rotation.x = polarity * Math.PI / 2
      vmps[[i, k, j]] = vmp
      this.anim_mats.push(vmp)
      this.group.add(vmp.group)
    })

    const { i: { size: isize }, j: { size: jsize } } = this.getBlockInfo()
    let curi = -1
    let curj = sweep ? -1 : 0

    this.getIndex = () => curi

    return () => {
      // update indexes
      const [oldi, oldj] = [curi, curj]
      sweep && (curj = (curj + 1) % jsize)
      curj == 0 && curi++

      // clear old input hilights
      if (oldi >= 0 && !this.params.anim['hide inputs']) {
        sweep && this.grid('j', ({ start: j, extent: jx }) => {
          oldj < jx && this.right.setColorsAndSizes(undefined, j + oldj)
        })
        oldi != curi && this.grid('i', ({ start: i, extent: ix }) => {
          oldi < ix && this.left.setColorsAndSizes(i + oldi, undefined)
        })
      }

      // end of cycle
      if (curi == isize) {
        this.onAnimDone()
        return
      }

      // start of cycle
      if (curi == 0 && curj == 0) {
        Object.values(vmps).forEach(vmp => vmp.setRowGuides())
        results.forEach(r => r.hide())
      }

      // new input hilights
      if (!this.params.anim['hide inputs']) {
        sweep && this.grid('j', ({ start: j, extent: jx }) => {
          curj < jx && this.right.bumpColor(undefined, j + curj)
        })
        oldi != curi && this.grid('i', ({ start: i, extent: ix }) => {
          curi < ix && this.left.bumpColor(i + curi, undefined)
        })
      }

      // update intermediates
      this.grid('ikj', (
        { start: i, extent: ix, index: ii },
        { start: k },
        { start: j, extent: jx, index: ji }
      ) => {
        const vmp = vmps[[i, k, j]]
        if (curi < ix && curj < jx) {
          util.updateProps(vmp.group.position, { x: j + (ji * gap) + curj, y: gap + i + (ii * gap) + curi })
          vmp.reinit((ki, ji) => this.ikjmul(i + curi, k + ki, j + curj + ji))
        }
      })

      // reveal new results
      this.grid('ij', ({ start: i, extent: ix }, { start: j, end: je, extent: jx }) => {
        curi < ix && curj < jx && results.forEach(r => r.show(i + curi, sweep ? j + curj : [j, je]))
      })

      // update labels
      this.updateLabels()
    }
  }

  getMvprodBump(sweep) {
    const { gap, polarity } = this.getLayoutInfo()
    const results = this.getAnimResultMats()

    const mvps = {}
    this.grid('ikj', (
      { start: i, extent: ix, index: ii },
      { start: k, extent: kx, index: ki },
      { start: j, index: ji },
    ) => {
      const mvpinit = (iii, kii) => this.ikjmul(i + iii, k + kii, j)
      const data = Array2D.fromInit(sweep ? 1 : ix, kx, mvpinit)
      const mvp = new Mat(data, this.getAnimIntermediateParams(this.params.name + `.mvp[${ii}, ${ki}, ${ji}]`), this.context, true)
      mvp.hide()
      const z = polarity < 0 ? this.getExtent().z - k - (gap * ki) : k + (gap * ki)
      util.updateProps(mvp.group.position, { x: gap + j + ji, y: i + ii, z })
      mvp.group.rotation.y = polarity * -Math.PI / 2
      mvps[[i, k, j]] = mvp
      this.anim_mats.push(mvp)
      this.group.add(mvp.group)
    })

    const { i: { size: isize }, j: { size: jsize } } = this.getBlockInfo()
    let curj = -1
    let curi = sweep ? -1 : 0

    this.getIndex = () => curj

    return () => {
      // update indexes
      const [oldi, oldj] = [curi, curj]
      sweep && (curi = (curi + 1) % isize)
      curi == 0 && curj++

      // clear old input hilights
      if (oldj >= 0 && !this.params.anim['hide inputs']) {
        sweep && this.grid('i', ({ start: i, extent: ix }) => {
          oldi < ix && this.left.setColorsAndSizes(i + oldi, undefined)
        })
        oldj != curj && this.grid('j', ({ start: j, extent: jx }) => {
          oldj < jx && this.right.setColorsAndSizes(undefined, j + oldj)
        })
      }

      // end of cycle
      if (curj == jsize) {
        this.onAnimDone()
        return
      }

      // start of cycle
      if (curj == 0 && curi == 0) {
        Object.values(mvps).forEach(vmp => vmp.setRowGuides())
        results.forEach(r => r.hide())
      }

      // new input hilights
      if (!this.params.anim['hide inputs']) {
        sweep && this.grid('i', ({ start: i, extent: ix }) => {
          curi < ix && this.left.bumpColor(i + curi, undefined)
        })
        oldj != curj && this.grid('j', ({ start: j, extent: jx }) => {
          curj < jx && this.right.bumpColor(undefined, j + curj)
        })
      }

      // update intermediates
      this.grid('ikj', (
        { start: i, extent: ix, index: ii },
        { start: k },
        { start: j, extent: jx, index: ji }
      ) => {
        const mvp = mvps[[i, k, j]]
        if (curi < ix && curj < jx) {
          util.updateProps(mvp.group.position, { x: gap + j + (ji * gap) + curj, y: i + (ii * gap) + curi })
          mvp.reinit((ii, ki) => this.ikjmul(i + curi + ii, k + ki, j + curj))
        }
      })

      // reveal new results
      this.grid('ij', ({ start: i, end: ie, extent: ix }, { start: j, extent: jx }) => {
        curi < ix && curj < jx && results.forEach(r => r.show(sweep ? i + curi : [i, ie], j + curj))
      })

      // update labels
      this.updateLabels()
    }
  }

  getVvprodBump(sweep) {
    const { gap, polarity } = this.getLayoutInfo()
    const { z: extz } = this.getExtent()
    // no intermediate result planes for vvprod, too cluttered. just sum it into final result
    const results = [this.result]

    // pre-epilog shadow for result accum
    const pre_epilog = Array2D.fromInit(this.H, this.W, () => 0)

    const vvps = {}
    this.grid('ikj', (
      { start: i, extent: ix, index: ii },
      { start: k, index: ki },
      { start: j, extent: jx, index: ji }
    ) => {
      const vvpinit = (iii, jii) => this.ikjmul(i + iii, k, j + jii)
      const data = Array2D.fromInit(ix, sweep ? 1 : jx, vvpinit)
      const vvp = new Mat(data, this.getAnimIntermediateParams(this.params.name + `.vvp[${ii}, ${ki}, ${ji}]`), this.context, true)
      vvp.hide()
      const z = polarity > 0 ? gap + k + ki : extz - gap - k - ki
      util.updateProps(vvp.group.position, { x: j + ji * gap, y: i + ii * gap, z })
      vvps[[i, k, j]] = vvp
      this.anim_mats.push(vvp)
      this.group.add(vvp.group)
    })

    const { k: { size: ksize }, j: { size: jsize } } = this.getBlockInfo()
    let curk = -1
    let curj = sweep ? -1 : 0

    this.getIndex = () => curk

    return () => {
      // update indexes
      const [oldk, oldj] = [curk, curj]
      curk++
      if (sweep && curk % ksize == 0) {
        curk = 0
        curj++
      }

      // clear old input highlights
      if (oldk >= 0 && !this.params.anim['hide inputs']) {
        sweep ?
          this.grid('kj', ({ start: k, extent: kx }, { start: j, extent: jx }) => {
            oldk < kx && oldj < jx && this.right.setColorsAndSizes(k + oldk, j + oldj)
          }) :
          this.grid('k', ({ start: k, extent: kx }) => {
            oldk < kx && this.right.setColorsAndSizes(k + oldk, undefined)
          })
        this.grid('k', ({ start: k, extent: kx }) => {
          oldk < kx && this.left.setColorsAndSizes(undefined, k + oldk)
        })
      }

      // end of cycle
      if (sweep ? curj == jsize : curk == ksize) {
        this.onAnimDone()
        return
      }

      // start of cycle
      if (curj == 0 && curk == 0) {
        Object.values(vvps).forEach(vvp => vvp.setRowGuides())
        results.forEach(r => r.hide())
      }

      // new input highlights
      if (!this.params.anim['hide inputs']) {
        sweep ?
          this.grid('kj', ({ start: k, extent: kx }, { start: j, extent: jx }) => {
            curk < kx && curj < jx && this.right.bumpColor(k + curk, j + curj)
          }) :
          this.grid('k', ({ start: k, extent: kx }) => {
            curk < kx && this.right.bumpColor(k + curk, undefined)
          })
        this.grid('k', ({ start: k, extent: kx }) => {
          curk < kx && this.left.bumpColor(undefined, k + curk)
        })
      }

      // update intermediates
      this.grid('ikj', (
        { start: i },
        { start: k, extent: kx, index: ki },
        { start: j, extent: jx, index: ji }
      ) => {
        const vvp = vvps[[i, k, j]]
        if (curk < kx && curj < jx) {
          const z = polarity > 0 ? gap + k + (ki * gap) + curk : extz - gap - k - (ki * gap) - curk
          util.updateProps(vvp.group.position, { x: j + ji * gap + curj, z })
          vvp.reinit((iii, jii) => this.ikjmul(i + iii, k + curk, j + curj + jii))
        }
      })

      // no intermediate result planes for vvprod, too cluttered. just sum it into final result
      // also we go thru some gymnastics to do epilog during sweep
      this.grid('kj', ({ start: k, extent: kx, index: ki }, { start: j, end: je, extent: jx }) => {
        if (curk < kx && curj < jx) {
          const running_dp = (ii, ji) => {
            const x = this.left.getData(ii, k + curk) * this.right.getData(k + curk, ji)
            return (ki == 0 && curk == 0) ? x : (pre_epilog.get(ii, ji) + x)
          }
          pre_epilog.reinit(running_dp, undefined, undefined, sweep ? j + curj : [j, je])

          const pw_epilog_dp = (ii, ji) => this.applyPointwiseEpilog(pre_epilog.get(ii, ji))
          results[0].reinit(pw_epilog_dp, undefined, undefined, sweep ? j + curj : [j, je])
        }
      })
      applyInPlaceEpilog_(results[0].data.data, results[0].H, results[0].W, this.params.epilog)
      if (sweep) {
        this.grid('kj', ({ extent: kx }, { start: j, end: je, extent: jx }) => {
          if (curk < kx && curj < jx) {
            results[0].reinit(() => 0, undefined, undefined, [j + curj + 1, je])
          }
        })
      }
      results[0].setColorsAndSizes()

      // update labels
      this.updateLabels()
    }
  }
}

//
// layout schemes
//

const layoutToBool = layout => ({
  pol: !!POLARITIES.indexOf(layout.polarity),
  left: !!LEFT_PLACEMENTS.indexOf(layout['left placement']),
  right: !!RIGHT_PLACEMENTS.indexOf(layout['right placement']),
  res: !!RESULT_PLACEMENTS.indexOf(layout['result placement'])
})

const boolToLayout = ({ pol, left, right, res }) => ({
  polarity: POLARITIES[+pol],
  'left placement': LEFT_PLACEMENTS[+left],
  'right placement': RIGHT_PLACEMENTS[+right],
  'result placement': RESULT_PLACEMENTS[+res]
})

export const LAYOUT_RULES = {
  'blocks': (left_child, { pol, left, right, res }) => ({
    pol: !pol,
    left: left_child ? pol != res : !left,
    right: left_child ? !right : pol != res,
    res: pol == (left_child ? left : right),
  }),
  'zigzag': (left_child, { pol, left, right, res }) => ({
    pol: !pol,
    left: left_child ? pol != res : left,
    right: left_child ? right : pol != res,
    res: pol == (left_child ? left : right),
  }),
  'wheel': (left_child, { pol, left, right, res }) => ({
    pol: pol,
    left: left,
    right: right,
    res: res
  }),
}

export const childLayout = (parent_layout, rule, left_child) =>
  boolToLayout(rule(left_child, layoutToBool(parent_layout)))

export function setLayoutScheme(params, scheme_name) {
  scheme_name = util.syncProp(params.layout, 'scheme', scheme_name)
  const rule = LAYOUT_RULES[scheme_name]
  function f(p) {
    if (p.left.matmul) {
      p.left.layout = childLayout(p.layout, rule, true)
      f(p.left)
    }
    if (p.right.matmul) {
      p.right.layout = childLayout(p.layout, rule, false)
      f(p.right)
    }
  }
  rule && f(params)
}

// 
// exprs
//

export const default_dims = { i: 32, j: 32, k: 32 }

export const default_epilog = 'none'

export const defaultLeft = () => ({
  name: 'L',
  matmul: false,
  h: default_dims.i,
  w: default_dims.j,
  init: 'row major',
  url: '',
  min: -1,
  max: 1,
  dropout: 0,
})

export const defaultRight = () => ({
  name: 'R',
  matmul: false,
  h: default_dims.j,
  w: default_dims.k,
  init: 'col major',
  url: '',
  min: -1,
  max: 1,
  dropout: 0,
})

export const defaultAnim = () => ({
  alg: 'inherit',
})

export const defaultBlock = () => ({
  'i blocks': 1,
  'k blocks': 1,
  'j blocks': 1,
})

export const defaultLayout = () => ({
  polarity: 'negative',
  'left placement': 'left',
  'right placement': 'top',
  'result placement': 'front',
})

// adjust tree to match a param node's i/k/j blocks
export function fixBlocks(p, anc, root) {
  const getInfo = (p, anc, root) => {
    const is_root = anc.length == 0
    const pp = !is_root && anc[0](root)
    const panc = !is_root && anc.slice(1)
    const is_left = pp && p == pp.left
    const is_right = pp && p == pp.right
    return { is_left, is_right, pp, panc }
  }

  // from a given p, set i all the way down
  const setib = (i, p) => {
    p.block['i blocks'] = i
    p.left.block && setib(i, p.left)
  }

  // from a given p, set j all the way down
  const setjb = (j, p) => {
    p.block['j blocks'] = j
    p.right.block && setjb(j, p.right)
  }

  // from a given p, set k all the way down
  const setkb = (k, p) => {
    p.block['k blocks'] = k
    p.left.block && setjb(k, p.left)
    p.right.block && setib(k, p.right)
  }

  // return p and setter for where your i starts
  const iroot = (p, anc, root) => {
    const { is_left, is_right, pp, panc } = getInfo(p, anc, root)
    return is_left ? iroot(pp, panc, root) : is_right ? { p: pp, f: setkb } : { p, f: setib }
  }

  // return p and setter for where your j starts
  const jroot = (p, anc, root) => {
    const { is_left, is_right, pp, panc } = getInfo(p, anc, root)
    return is_right ? jroot(pp, panc, root) : is_left ? { p: pp, f: setkb } : { p, f: setjb }
  }

  const ir = iroot(p, anc, root)
  ir.f(p.block['i blocks'], ir.p)

  const jr = jroot(p, anc, root)
  jr.f(p.block['j blocks'], jr.p)

  // k always starts here
  setkb(p.block['k blocks'], p)
}

// adjust surroundings to match a param node's h/w
export function fixShape(h, w, p, anc, root) {
  const height = p => p.left ? height(p.left) : p.h
  const width = p => p.right ? width(p.right) : p.w

  const seth = (p, h) => p.left ? seth(p.left, h) : (p.h = h)
  const setw = (p, w) => p.right ? setw(p.right, w) : (p.w = w)

  const pp = anc[0](root)
  p === pp.left ? seth(pp.right, w) : setw(pp.left, h)
  anc.length > 1 && fixShape(height(pp.left), width(pp.right), pp, anc.slice(1), root)
}

export const leftLeaf = p => p.left.matmul ? leftLeaf(p.left) : p.left
export const rightLeaf = p => p.right.matmul ? rightLeaf(p.right) : p.right

// parseExpr, syncExpr

function parseExpr(s) {
  try {
    const node = spec => typeof spec == 'string' ? { name: spec } : make(spec)
    const make = spec => {
      const i = spec[1] == '=' ? 2 : 0
      const rname = r => /\s+/.test(r.name) ? '(' + r.name + ')' : r.name
      const f = (left, x) => {
        const right = node(x)
        return { left, right, name: left.name + ' @ ' + rname(right) }
      }
      const p = spec.slice(i + 1).reduce(f, node(spec[i]))
      i > 0 && (p.name = spec[0])
      return p
    }
    s = '[' + s.replace(/\s+/g, '').
      replace(/(\w+[\w\.\-\!\#\$\%\^\&\/\[\]]*)/g, '"$1"').
      replaceAll('@', ',').
      replaceAll('(', '[').
      replaceAll(')', ']').
      replaceAll('=', ',"=",') + ']'
    let spec = eval?.(s)
    while (spec.length == 1) {
      spec = spec[0]
    }
    return make(spec)
  } catch (e) {
    console.log(`error evaluating '${s}': ${e.message}`)
  }
}

export function syncExpr(params) {
  if (params.expr == genExpr(params)) {
    return true
  }

  const foundParams = {}

  const findParams = (p, n) => p.name == n ?
    (foundParams[p.name] = p) :
    (p.left && findParams(p.left, n)) ||
    (p.right && findParams(p.right, n)) ||
    undefined

  const childParams = (p, is_left) => {
    const found = findParams(params, p.name)
    if (p.left && p.right) {
      if (found && found.left && found.right) {
        return {
          ...util.copyTree(found),
          left: childParams(p.left, true),
          right: childParams(p.right, false),
          matmul: true,
        }
      } else {
        const cp = {
          epilog: default_epilog,
          anim: defaultAnim(),
          block: defaultBlock(),
          layout: defaultLayout(),
          left: childParams(p.left, true),
          right: childParams(p.right, false),
          name: p.name,
          matmul: true,
        }
        if (found) {
          leftLeaf(cp).h = found.h
          rightLeaf(cp).w = found.w
        }
        return cp
      }
    } else {
      if (found) {
        return !(found.left && found.right) ? util.copyTree(found) : {
          ...(is_left ? leftLeaf(found) : rightLeaf(found)),
          w: rightLeaf(found).w,
          name: p.name,
          matmul: false,
        }
      }
      return {
        ...(is_left ? defaultLeft() : defaultRight()),
        name: p.name,
        matmul: false,
      }
    }
  }

  const fixShapes = (p, anc = [p => p]) => {
    if (p.left && p.right) {
      const path = anc[0]
      if (!foundParams[p.right.name]) {
        fixShapes(p.left, [p => path(p).left].concat(anc))
        fixShapes(p.right, [p => path(p).right].concat(anc))
      } else {
        fixShapes(p.right, [p => path(p).right].concat(anc))
        fixShapes(p.left, [p => path(p).left].concat(anc))
      }
    } else {
      fixShape(p.h, p.w, p, anc.slice(1), new_params)
    }
  }

  const p = parseExpr(params.expr)
  if (!p) {
    return false
  }

  const new_params = {
    name: p.name,
    left: childParams(p.left, true),
    right: childParams(p.right, false)
  }

  fixShapes(new_params)
  util.updateProps(params, new_params)
  setLayoutScheme(params)

  return true
}

export function genExpr(p) {
  const passign = e => /^\w+\s+=/.test(e) ? `(${e})` : e
  const l = p.left.matmul ? passign(genExpr(p.left)) : p.left.name
  const r = p.right.matmul ? '(' + genExpr(p.right) + ')' : p.right.name
  const expanded = `${l} @ ${r}`
  const named = `${p.left.name} @ ${p.right.name}`
  return p.name == expanded || p.name == named ? expanded : `${p.name} = ${expanded}`
}


