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

export const INIT_FUNCS = {
  rows: (i, j, h) => h > 1 ? i / (h - 1) : 0,
  cols: (i, j, h, w) => w > 1 ? j / (w - 1) : 0,
  'row major': (i, j, h, w) => h * w > 1 ? (i * w + j) / (h * w - 1) : 0,
  'col major': (i, j, h, w) => h * w > 1 ? (j * h + i) / (h * w) : 0,
  'pt linear': (i, j, h, w) => (2 * Math.random() - 1) / Math.sqrt(w),
  'pt linear+': (i, j, h, w) => Math.max((2 * Math.random() - 1) / Math.sqrt(w)),
  uniform: () => Math.random(),
  gaussian: () => gaussianRandom(0.5, 0.5),
  tril: (i, j) => j <= i ? 1 : 0,
  triu: (i, j) => j >= i ? 1 : 0,
  eye: (i, j) => i == j ? 1 : 0,
  diff: (i, j) => i == j ? 1 : i == j + 1 ? -1 : 0,
}

export const INITS = Object.keys(INIT_FUNCS)

const USE_RANGE = ['rows', 'cols', 'row major', 'col major', 'uniform', 'gaussian']

export function useRange(name) {
  return USE_RANGE.indexOf(name) >= 0
}

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

export const EPILOGS = ['none', 'x/J', 'x/sqrt(J)', 'softmax(x/sqrt(J))', 'tanh', 'relu', 'layernorm']


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
      const x = init(i, j, h, w)
      if (isNaN(x)) {
        throw Error(`HEY init(${i}, ${j}, ${h}, ${w}) is NaN`)
      }
      data[ptr] = x
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
// Mat
//

const ELEM_SIZE = 2000
const ZERO_COLOR = new THREE.Color(0, 0, 0)
const COLOR_TEMP = new THREE.Color()

function emptyPoints(h, w) {
  const n = h * w
  const points = new Float32Array(n * 3)
  for (let i = 0, ptr = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      points[ptr++] = j
      points[ptr++] = i
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

  constructor(data, params, init_vis) {
    this.params = { ...params }

    this.data = data
    this.H = data.h
    this.W = data.w
    this.absmax = this.data.absmax()

    this.points = emptyPoints(this.H, this.W)
    this.points.name = `${this.params.name}.points`

    if (init_vis) {
      this.initVis()
    }
  }

  initVis() {
    this.setColorsAndSizes()

    this.inner_group = new THREE.Group()
    this.inner_group.name = `${this.params.name}.inner_group`
    this.inner_group.add(this.points)

    const gap = this.params.gap
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
    return this._extents || (this._extents = {
      x: this.W + 2 * this.params.gap - 1,
      y: this.H + 2 * this.params.gap - 1,
      z: 0,
    })
  }

  sizeFromData(x) {
    if (x === undefined || isNaN(x)) {
      throw Error(`HEY sizeFromData(${x})`)
    }

    const local_sens = this.params.sensitivity == 'local'
    const absx = Math.abs(x)
    const absmax = local_sens ? this.absmax : this.getGlobalAbsmax()

    if (absx === Infinity) {
      return ELEM_SIZE
    }

    const vol = absmax == 0 ? 0 : absx / absmax
    const zsize = this.params['min size'] * ELEM_SIZE
    const size = zsize + (ELEM_SIZE - zsize) * Math.cbrt(vol)

    if (absx > absmax || size < 0 || size > ELEM_SIZE || isNaN(size)) {
      throw Error(`HEY x ${x} size ${size} absx ${absx} absmax ${absmax} zsize ${zsize} sens ${local_sens}`)
    }

    return size
  }

  colorFromData(x) {
    if (x === undefined || isNaN(x)) {
      throw Error(`HEY colorFromData(${x})`)
    }

    const local_sens = this.params.sensitivity == 'local'
    const absx = Math.abs(x)
    const absmax = local_sens ? this.absmax : this.getGlobalAbsmax()

    if (absx === Infinity) {
      return COLOR_TEMP.setHSL(1.0, 1.0, 1.0)
    }

    const hue_vol = absmax == 0 ? 0 : x / absmax
    const gap = this.params['hue gap'] * Math.sign(x)
    const hue = (this.params['zero hue'] + gap + (Math.cbrt(hue_vol) * this.params['hue spread'])) % 1

    const light_vol = absmax == 0 ? 0 : absx / absmax
    const range = this.params['max light'] - this.params['min light']
    const light = this.params['min light'] + range * Math.cbrt(light_vol)

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
    const c = this.group.localToWorld(new THREE.Vector3()).sub(this.params.camera.position).normalize()
    const m = this.group.getWorldDirection(new THREE.Vector3())
    return m.angleTo(c) < Math.PI / 2
  }

  isRightSideUp() {
    const q = new THREE.Quaternion()
    const p = new THREE.Vector3(0, -1, 0).applyQuaternion(this.group.getWorldQuaternion(q))
    const c = new THREE.Vector3(0, 1, 0).applyQuaternion(this.params.camera.quaternion)
    return p.angleTo(c) < Math.PI / 2
  }

  setRowGuides(light) {
    light = util.syncProp(this.params, 'row guides', light)
    if (this.row_guide_group) {
      this.inner_group.remove(this.row_guide_group)
      util.disposeAndClear(this.row_guide_group)
    }
    if (light > 0.0) {
      this.row_guide_group = util.rowGuide(this.H, this.W, light)
      this.inner_group.add(this.row_guide_group)
    }
  }

  setFlowGuide(light) { }

  setLegends(enabled) {
    enabled = util.syncProp(this.params, 'legends', enabled)
    if (this.name_text) {
      this.inner_group.remove(this.name_text)
      util.disposeAndClear(this.name_text)
    }
    if (enabled && this.params.name) {
      const color = 0xCCCCFF
      const size = Math.cbrt(Math.max(5, this.H) * Math.max(this.W, 5)) / 2
      const facing = this.isFacing()
      const rsu = this.isRightSideUp()
      let suf = this.params.tag ? ` (${this.params.tag}` : ''
      this.name_text = this.params.getText(this.params.name + suf, color, size)
      this.name_text.name = `${this.params.name}.name`
      const { h, w } = util.bbhw(this.name_text.geometry)
      this.name_text.geometry.rotateZ(rsu ? Math.PI : 0)
      this.name_text.geometry.rotateY(facing ? Math.PI : 0)
      const xdir = facing == rsu ? 1 : -1
      const ydir = rsu ? 1 : 0
      const zdir = facing ? 1 : -1
      this.name_text.geometry.translate(
        util.center(this.W - 1, xdir * w),
        ydir * h + util.center(this.H - 1, h),
        -zdir * h / 2
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
    spotlight = util.syncProp(this.params, 'spotlight', spotlight)
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
      this.params.raycaster.params.Points.threshold = spotlight
      const intersects = this.params.raycaster.intersectObject(this.points)
      intersects.forEach(x => {
        const index = x.index
        const i = Math.floor(index / this.W)
        const j = index % this.W
        if (!this.isHidden(i, j)) {
          const x = this.getData(i, j)
          if (x != 0) { // declutter
            let label = this.label_cache[index]
            const facing = this.isFacing()
            const rsu = this.isRightSideUp()
            if (!label || label.facing != facing || label.rsu != rsu) {
              const fsiz = 0.16 - 0.008 * Math.log10(Math.floor(1 + Math.abs(x)))
              label = this.params.getText(`${x.toFixed(4)}`, 0xffffff, fsiz)
              label.name = `${this.params.name}.label[${i}, ${j}]`
              label.value = x
              label.facing = facing
              label.rsu = rsu
              const zdir = facing ? 1 : -1
              label.geometry.rotateX(zdir * Math.PI)
              label.geometry.rotateY(facing ? 0 : Math.PI)
              label.geometry.rotateZ(rsu ? 0 : Math.PI)
              const { h, w } = util.bbhw(label.geometry)
              label.geometry.translate(
                util.center(j * 2, (rsu ? zdir : -zdir) * w),
                h + util.center(i * 2, h),
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
export const SENSITIVITIES = ['local', 'global']
export const ANIM_ALGS = ['none', 'dotprod (row major)', 'dotprod (col major)', 'axpy', 'vmprod', 'mvprod', 'vvprod']


export class MatMul {

  constructor(params, init_vis = true) {
    this.params = { ...params }
    this.group = new THREE.Group()
    this.group.name = `${this.params['result name']}.group`

    this.H = params.I
    this.D = params.J
    this.W = params.K

    this.initLeft()
    this.initRight()
    this.initResult()

    if (init_vis) {
      this.initVis()
    }
  }

  disposeAll() {
    util.disposeAndClear(this.group)
  }

  prepChildParams(params) {
    params.getGlobalAbsmax = this.getGlobalAbsmax.bind(this)
    return params
  }

  getLeafParams() {
    // TODO transfer only what's needed from parent
    return {
      ...this.params,
      getGlobalAbsmax: this.getGlobalAbsmax.bind(this),
      depth: this.params.depth + 1,
      max_depth: this.params.depth + 1,
      height: 0,
      count: 0,
    }
  }

  initLeft() {
    if (this.params.left_mm) {
      this.left = new MatMul(this.prepChildParams(this.params.left_mm), false)
    } else {
      const data = this.params.left_data || (_ => {
        const init = this.params['left init']
        const min = this.params['left min']
        const max = this.params['left max']
        const dropout = this.params['left dropout']
        const f = getInitFunc(init, min, max, dropout)
        return Array2D.fromInit(this.H, this.D, f)
      })()
      const params = { ...this.getLeafParams(), name: this.params['left name'] }
      this.left = new Mat(data, params, false)
    }
  }

  initRight() {
    if (this.params.right_mm) {
      this.right = new MatMul(this.prepChildParams(this.params.right_mm), false)
    } else {
      const data = this.params.right_data || (_ => {
        const name = this.params['right init']
        const min = this.params['right min']
        const max = this.params['right max']
        const dropout = this.params['right dropout']
        const f = getInitFunc(name, min, max, dropout)
        return Array2D.fromInit(this.D, this.W, f)
      })()
      const params = { ...this.getLeafParams(), name: this.params['right name'] }
      this.right = new Mat(data, params, false)
    }
  }

  initResult() {
    const result_init = (i, j) => this.dotprod_val(i, j)
    const data = Array2D.fromInit(this.H, this.W, result_init, this.params.epilog)
    const params = {
      ...this.getLeafParams(),
      name: this.params['result name'],
      // TODO clean up
      height: this.params.height,
      depth: this.params.depth,
      max_depth: this.params.max_depth,
      count: this.params.count
    }
    this.result = new Mat(data, params, false)
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
      x += this.left.getData(i, j) * this.right.getData(j, k)
    }
    if (isNaN(x)) {
      throw Error(`HEY dotprod_val(${i}, ${k}, ${minj}, ${maxj}) is NaN`)
    }
    const epi = this.params.epilog
    return epi == 'x/J' ? x / this.D :
      epi == 'x/sqrt(J)' || epi == 'softmax(x/sqrt(J))' ? x / Math.sqrt(this.D) :
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
    return this._extents || (this._extents = {
      x: this.W + 2 * this.params.gap - 1,
      y: this.H + 2 * this.params.gap - 1,
      z: this.D + 2 * this.params.gap - 1,
    })
  }

  initVis(params = undefined) {
    if (params) {
      this.params = { ...params }
    }

    util.disposeAndClear(this.group)
    this.flow_guide_group = undefined
    this.anim_mats = []

    function next(x) {
      const sep = x.indexOf('/')
      return sep == -1 ? x : x.slice(sep + 1) + '/' + x.slice(0, sep)
    }

    this.left.params.polarity = next(this.params.polarity)
    this.left.params['left placement'] = next(this.params['left placement'])
    this.left.params['right placement'] = next(this.params['right placement'])
    this.left.params['result placement'] = next(this.params['result placement'])

    this.right.params.polarity = next(this.params.polarity)
    this.right.params['left placement'] = next(this.params['left placement'])
    this.right.params['right placement'] = next(this.params['right placement'])
    this.right.params['result placement'] = next(this.params['result placement'])

    this.initLeftVis()
    this.initRightVis()
    this.initResultVis()

    this.setFlowGuide()
    this.setRowGuides()
  }

  initLeftVis() {
    this.left.initVis()
    if (this.params.polarity.startsWith('positive')) {
      this.left.group.rotation.y = -Math.PI / 2
      this.left.group.position.x = this.params['left placement'].startsWith('left') ?
        -this.getLeftScatter() :
        this.getExtent().x + this.left.getExtent().z + this.getLeftScatter()
    } else { // negative
      this.left.group.rotation.y = Math.PI / 2
      this.left.group.position.z = this.getExtent().z
      this.left.group.position.x = this.params['left placement'].startsWith('left') ?
        -(this.left.getExtent().z + this.getLeftScatter()) :
        this.getExtent().x + this.getLeftScatter()
    }
    this.group.add(this.left.group)
  }

  initRightVis() {
    this.right.initVis()
    if (this.params.polarity.startsWith('positive')) {
      this.right.group.rotation.x = Math.PI / 2
      this.right.group.position.y = this.params['right placement'].startsWith('top') ?
        -this.getRightScatter() :
        this.getExtent().y + this.right.getExtent().z + this.getRightScatter()
    } else { // negative
      this.right.group.rotation.x = -Math.PI / 2
      this.right.group.position.z = this.getExtent().z
      this.right.group.position.y = this.params['right placement'].startsWith('top') ?
        -(this.right.getExtent().z + this.getRightScatter()) :
        this.getExtent().y + this.getRightScatter()
    }
    this.group.add(this.right.group)
  }

  initResultVis() {
    this.result.initVis()
    this.result.group.position.z = this.params['result placement'].startsWith('back') ?
      this.getExtent().z :
      0
    this.group.add(this.result.group)
  }

  getPlacementInfo() {
    return {
      polarity: this.params.polarity.startsWith('positive') ? 1 : -1,
      left: this.params['left placement'].startsWith('left') ? 1 : -1,
      right: this.params['right placement'].startsWith('top') ? 1 : -1,
      result: this.params['result placement'].startsWith('front') ? 1 : -1,
      gap: this.params.gap,
      left_scatter: this.getLeftScatter(),
      right_scatter: this.getRightScatter(),
    }
  }

  setFlowGuide(light = undefined) {
    if (light != this.params['flow guides']) {
      light = util.syncProp(this.params, 'flow guides', light)
      if (this.flow_guide_group) {
        this.group.remove(this.flow_guide_group)
        util.disposeAndClear(this.flow_guide_group)
        this.flow_guide_group = undefined
      }
      if (light > 0.0) {
        this.flow_guide_group = util.flowGuide(this.H, this.D, this.W, this.getPlacementInfo(), light)
        this.group.add(this.flow_guide_group)
      }
    }
    this.left.setFlowGuide(light)
    this.right.setFlowGuide(light)
  }

  scatterFromCount(count) {
    const blast = (count >= this.params.molecule) ? count ** this.params.blast : 0
    return this.params.scatter * blast
  }

  getLeftScatter() {
    return this.scatterFromCount(this.left.params.count)
  }

  getRightScatter() {
    return this.scatterFromCount(this.right.params.count)
  }

  updateLabels(params = undefined) {
    if (params) {
      this.params.spotlight = params.spotlight
      this.params['interior spotlight'] = params['interior spotlight']
    }

    const spotlight = this.params.spotlight
    this.left.updateLabels(spotlight)
    this.right.updateLabels(spotlight)
    this.result.updateLabels(spotlight)

    const interior_spotlight = this.params['interior spotlight'] ? spotlight : 0
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
    util.syncProp(this.params, 'hide inputs', hide)
    if (this.params.left_mm) {
      this.left.hideInputs(hide)
    } else if (this.params.alg != 'none') {
      hide ? this.left.hide() : this.left.show()
    }
    if (this.params.right_mm) {
      this.right.hideInputs(hide)
    } else if (this.params.alg != 'none') {
      hide ? this.right.hide() : this.right.show()
    }
  }

  setRowGuides(light) {
    light = util.syncProp(this.params, 'row guides', light)
    this.left.setRowGuides(light)
    this.right.setRowGuides(light)
    this.result.setRowGuides(light)
  }

  setLegends(enabled = undefined) {
    util.syncProp(this.params, 'legends', enabled)
    this.left.setLegends(enabled)
    this.right.setLegends(enabled)
    this.result.setLegends(enabled)
  }

  // animation

  initAnimation(cb) {
    if (this.params.alg == 'none') {
      if (this.params['hide inputs']) {
        !this.left_mm && this.left.show()
        !this.right_mm && this.right.show()
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

    let left_done = true, right_done = true

    this.alg_join = () => {
      const alg = this.params.alg

      const lalg = this.params.left_mm && !left_done ?
        (this.left.getIndex() == this.getIndex() ? this.left.alg_join() : 'mixed') :
        'none'

      const ralg = this.params.right_mm && !right_done ?
        (this.right.getIndex() == this.getIndex() ? this.right.alg_join() : 'mixed') :
        'none'

      return (lalg == 'none' && ralg == 'none') ? alg :
        (ralg == 'none') ?
          (lalg == 'vmprod' && alg == 'vmprod' ? 'vmprod' :
            lalg == 'mvprod' && alg == 'vvprod' ? 'vvprod' :
              'mixed') :
          (lalg == 'none') ?
            (ralg == 'mvprod' && alg == 'mvprod' ? 'mvprod' :
              ralg == 'vmprod' && alg == 'vvprod' ? 'vvprod' :
                'mixed') :
            'mixed'
    }

    const can_fuse = () => this.alg_join() != 'mixed'

    const start = () => {
      const result_bump = bumps[this.params.alg]()

      this.bump = () => {
        const go = left_done && right_done || this.params.fuse && can_fuse()
        left_done || this.left.bump()
        right_done || this.right.bump()
        go && result_bump()
      }

      if (this.params.left_mm) {
        left_done = false
        this.left.initAnimation(() => left_done = true)
      }
      if (this.params.right_mm) {
        right_done = false
        this.right.initAnimation(() => right_done = true)
      }

      if (this.params['hide inputs']) {
        this.left.hide()
        this.right.hide()
      }
      this.result.hide()
    }

    this.onAnimDone = () => {
      this.clearAnimMats()
      cb ? cb() : start()
    }

    start()
  }

  getBlockInfo() {
    const ni = Math.min(this.params['i blocks'], this.H)
    const nj = Math.min(this.params['j blocks'], this.D)
    const nk = Math.min(this.params['k blocks'], this.W)
    return {
      i: { n: ni, size: Math.ceil(this.H / ni), max: this.H },
      j: { n: nj, size: Math.ceil(this.D / nj), max: this.D },
      k: { n: nk, size: Math.ceil(this.W / nk), max: this.W },
    }
  }

  grid(dims, f) {
    const info = this.getBlockInfo()
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

  getAnimMatParams() {
    const params = { ...this.getLeafParams() }
    params.sensitivity = 'local'
    params.stretch_absmax = true
    return params
  }

  clearAnimMats() {
    this.anim_mats.map(m => {
      this.group.remove(m.group)
      util.disposeAndClear(m.group)
    })
    this.anim_mats = []
  }

  getAnimResultMats() {
    if (this.getBlockInfo().j.n == 1) {
      this.result.params.stretch_absmax = true
      return [this.result]
    }
    const { gap, polarity } = this.getPlacementInfo()
    const { z: extz } = this.getExtent()
    const results = []
    this.grid('j', ({ start: j, end: je }) => {
      const result_init = (i, k) => this.dotprod_val(i, k, j, je)
      const data = Array2D.fromInit(this.H, this.W, result_init)
      const result = new Mat(data, this.getAnimMatParams(), true)
      result.group.position.z = polarity > 0 ? j + gap : extz - je + 1 - gap
      result.hide()
      results.push(result)
      this.group.add(result.group)
      this.anim_mats.push(result)
    })
    return results
  }

  getVmprodBump(sweep) {
    const { gap, polarity } = this.getPlacementInfo()
    const results = this.getAnimResultMats()

    const vmps = {}
    this.grid('ijk', ({ start: i }, { start: j, extent: jx }, { start: k, extent: kx }) => {
      const vmpinit = (ji, ki) => this.ijkmul(i, j + ji, k + ki)
      const data = Array2D.fromInit(jx, sweep ? 1 : kx, vmpinit)
      const vmp = new Mat(data, this.getAnimMatParams(), true)
      vmp.hide()
      const z = polarity < 0 ? this.getExtent().z - j : j
      util.updateProps(vmp.group.position, { x: k, y: gap + i, z })
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
      if (oldi >= 0 && !this.params['hide inputs']) {
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
      curi == 0 && curk == 0 && results.forEach(r => r.hide())

      // new input hilights
      if (!this.params['hide inputs']) {
        sweep && this.grid('k', ({ start: k, extent: kx }) => {
          curk < kx && this.right.bumpColor(undefined, k + curk)
        })
        oldi != curi && this.grid('i', ({ start: i, extent: ix }) => {
          curi < ix && this.left.bumpColor(i + curi, undefined)
        })
      }

      // update intermediates
      this.grid('ijk', ({ start: i, extent: ix }, { start: j }, { start: k, extent: kx }) => {
        const vmp = vmps[[i, j, k]]
        if (curi < ix && curk < kx) {
          util.updateProps(vmp.group.position, { x: k + curk, y: gap + i + curi })
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
    this.grid('ijk', ({ start: i, extent: ix }, { start: j, extent: jx }, { start: k }) => {
      const mvpinit = (ii, ji) => this.ijkmul(i + ii, j + ji, k)
      const data = Array2D.fromInit(sweep ? 1 : ix, jx, mvpinit)
      const mvp = new Mat(data, this.getAnimMatParams(), true)
      mvp.hide()
      const z = polarity < 0 ? this.getExtent().z - j : j
      util.updateProps(mvp.group.position, { x: gap + k, y: i, z })
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
      if (oldk >= 0 && !this.params['hide inputs']) {
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
      curk == 0 && curi == 0 && results.forEach(r => r.hide())

      // new input hilights
      if (!this.params['hide inputs']) {
        sweep && this.grid('i', ({ start: i, extent: ix }) => {
          curi < ix && this.left.bumpColor(i + curi, undefined)
        })
        oldk != curk && this.grid('k', ({ start: k, extent: kx }) => {
          curk < kx && this.right.bumpColor(undefined, k + curk)
        })
      }

      // update intermediates
      this.grid('ijk', ({ start: i, extent: ix }, { start: j }, { start: k, extent: kx }) => {
        const mvp = mvps[[i, j, k]]
        if (curi < ix && curk < kx) {
          util.updateProps(mvp.group.position, { x: gap + k + curk, y: i + curi })
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
    this.grid('ijk', ({ start: i, extent: ix }, { start: j }, { start: k, extent: kx }) => {
      const vvpinit = (ii, ki) => this.ijkmul(i + ii, j, k + ki)
      const data = Array2D.fromInit(ix, sweep ? 1 : kx, vvpinit)
      const vvp = new Mat(data, this.getAnimMatParams(), true)
      vvp.hide()
      const z = polarity > 0 ? gap + j : extz - gap - j
      util.updateProps(vvp.group.position, { x: k, y: i, z })
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
      if (oldj >= 0 && !this.params['hide inputs']) {
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
      curk == 0 && curj == 0 && results.forEach(r => r.hide())

      // new input highlights
      if (oldj >= 0 && !this.params['hide inputs']) {
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
      this.grid('ijk', ({ start: i }, { start: j, extent: jx, end: je }, { start: k, end: ke, extent: kx }) => {
        const vvp = vvps[[i, j, k]]
        if (curj < jx && curk < kx) {
          const z = polarity > 0 ? gap + j + curj : extz - gap - j - curj
          util.updateProps(vvp.group.position, { x: k + curk, z })
          vvp.reinit((ii, ki) => this.ijkmul(i + ii, j + curj, k + curk + ki))
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

