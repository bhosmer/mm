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

const USE_RANGE = ['rows', 'cols', 'row major', 'col major', 'uniform', 'gaussian']

function useRange(name) {
  return USE_RANGE.indexOf(name) >= 0
}

function getInitFunc(name, min = 0, max = 1, sparsity = 0) {
  const f = INIT_FUNCS[name]
  if (!f) {
    throw Error(`unrecognized initializer ${name}`)
  }
  const scaled = useRange(name) && (min != 0 || max != 1) ?
    (i, j, h, w) => min + Math.max(0, max - min) * f(i, j, h, w) :
    f
  const sparse = sparsity > 0 ?
    (i, j, h, w) => Math.random() > sparsity ? scaled(i, j, h, w) : 0 :
    scaled
  return sparse
}

function initFuncFromParams(init_params) {
  const { name, min, max, sparsity } = init_params
  return getInitFunc(name, min, max, sparsity)
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
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(points, 3));
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

    if (init_vis) {
      this.initVis()
    }
  }

  initVis() {
    this.setColorsAndSizes()

    this.inner_group = new THREE.Group()
    this.inner_group.add(this.points)

    const gap = this.params.gap
    util.updateProps(this.inner_group.position, { x: gap, y: gap })

    this.group = new THREE.Group()
    this.group.add(this.inner_group)
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

    const hue_vol = absmax == 0 ? 0 : x / absmax
    const gap = this.params['hue gap'] * Math.sign(x)
    const hue = (this.params['zero hue'] + gap + (Math.cbrt(hue_vol) * this.params['hue spread'])) % 1

    const light_vol = absmax == 0 ? 0 : absx / absmax
    const range = this.params['max light'] - this.params['min light']
    const light = this.params['min light'] + range * Math.cbrt(light_vol)

    return new THREE.Color().setHSL(hue, 1.0, light)
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
    const c = new THREE.Color()
    return c.fromArray(this.points.geometry.attributes.pointColor.array, this.data.addr(i, j) * 3)
  }

  setColor(i, j, c) {
    c.toArray(this.points.geometry.attributes.pointColor.array, this.data.addr(i, j) * 3)
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
    const bump = new THREE.Color(0x808080)
    this.setColorsAndSizes(r, c, undefined, x => this.colorFromData(x).add(bump))
  }

  setRowGuides(enabled = undefined) {
    enabled = util.syncProp(this.params, 'row guides', enabled)
    if (enabled) {
      if (!this.row_guide_group) {
        this.row_guide_group = util.rowGuide(this.H, this.W)
        this.inner_group.add(this.row_guide_group)
      }
    } else {
      if (this.row_guide_group) {
        this.inner_group.remove(this.row_guide_group)
        this.row_guide_group.clear()
        this.row_guide_group = undefined
      }
    }
  }

  setFlowGuide(enabled) { }

  getLegendTextProps() {
    const sa_geo = Math.cbrt(Math.max(5, this.H) * Math.max(this.W, 5))
    return {
      name_color: 0xccccff,
      name_size: sa_geo / 2,
      dim_color: 0x00aaff,
      dim_size: sa_geo / 8,
    }
  }

  setLegends(enabled, props) {
    if (enabled) {
      if (this.legends_group) {
        this.inner_group.remove(this.legends_group)
        this.legends_group.clear()
      }
      this.legends_group = new THREE.Group()
      props = { ...this.getLegendTextProps(), ...props }
      if (props.name) {
        let suf = ''
        // suf += ` (${this.params.depth},${this.params.max_depth},${this.params.height})`
        // suf += this.params.count > 0 ? ` (${this.params.count})` : ''
        // suf += this.params.tag ? ` (${this.params.tag})` : ''
        const name = this.params.getText(props.name + suf, props.name_color, props.name_size)
        const { h, w } = util.bbhw(name.geometry)
        name.geometry.rotateZ(Math.PI)
        name.geometry.rotateY(Math.PI)
        name.geometry.translate(util.center(this.W - 1, w), h + util.center(this.H - 1, h), -(1 + h / 2))
        this.legends_group.add(name)
      }
      if (props.height) {
        const height = this.params.getText(`${props.height} = ${this.H}`, props.dim_color, props.dim_size)
        const { h, w } = util.bbhw(height.geometry)
        height.geometry.rotateX(Math.PI)
        const zrot = (props.hleft ? -1 : 1) * Math.PI / 2
        height.geometry.rotateZ(zrot)
        const spacer = 0.5
        const xoff = props.hleft ? -h * 1 - spacer : this.W - 1 + h + spacer
        const yoff = props.hleft ? w + util.center(this.H - 1, w) : util.center(this.H - 1, w)
        height.geometry.translate(xoff, yoff, 0)
        this.legends_group.add(height)
      }
      if (props.width) {
        const width = this.params.getText(`${props.width} = ${this.W}`, props.dim_color, props.dim_size)
        const { h, w } = util.bbhw(width.geometry)
        width.geometry.rotateX(Math.PI)
        const spacer = 0.5
        const xoff = util.center(this.W - 1, w)
        const yoff = props.wtop ? -h * 1 - spacer : this.H - 1 + h * 1.5 + spacer
        width.geometry.translate(xoff, yoff, 0)
        this.legends_group.add(width)
      }
      this.inner_group.add(this.legends_group)
    } else {
      if (this.legends_group) {
        this.inner_group.remove(this.legends_group)
        this.legends_group.clear()
        this.legends_group = undefined
      }
    }
  }

  checkLabel(i, j, x) {
    if (this.label_cache) {
      const addr = this.data.addr(i, j)
      const label = this.label_cache[addr]
      if (label != undefined && label.value != x) {
        this.label_cache[addr] = undefined
      }
    }
  }

  updateLabels(spotlight = undefined) {
    spotlight = util.syncProp(this.params, 'spotlight', spotlight)

    if (spotlight == 0) {
      if (this.label_group) {
        this.label_group.clear()
        this.inner_group.remove(this.label_group)
        this.label_group = undefined
      }
      return
    }

    if (!this.label_group) {
      this.label_group = new THREE.Group()
      this.inner_group.add(this.label_group)
      this.label_cache = []
    } else {
      this.label_group.clear()
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
          if (!label) {
            const fsiz = 0.16 - 0.008 * Math.log10(Math.floor(1 + Math.abs(x)))
            label = this.params.getText(`${x.toFixed(4)}`, 0xffffff, fsiz)
            label.value = x
            label.geometry.rotateX(Math.PI)
            const { h, w } = util.bbhw(label.geometry)
            label.geometry.translate(util.center(j * 2, w), h + util.center(i * 2, h), -0.25)
            this.label_cache[index] = label
          }
          this.label_group.add(label)
        }
      }
    })
  }
}

//
// MatMul
//

export class MatMul {

  constructor(params, init_vis = true) {
    this.params = { ...params }
    this.group = new THREE.Group()

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
        const sparsity = this.params['left sparsity']
        const f = getInitFunc(init, min, max, sparsity)
        return Array2D.fromInit(this.H, this.D, f)
      })()
      this.left = new Mat(data, this.getLeafParams(), false)
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
        const sparsity = this.params['right sparsity']
        const f = getInitFunc(name, min, max, sparsity)
        return Array2D.fromInit(this.D, this.W, f)
      })()
      this.right = new Mat(data, this.getLeafParams(), false)
    }
  }

  initResult() {
    const result_init = (i, j) => this.dotprod_val(i, j)
    const data = Array2D.fromInit(this.H, this.W, result_init, this.params.epilog)
    const params = this.getLeafParams()
    params.height = this.params.height
    params.depth = this.params.depth
    params.max_depth = this.params.max_depth
    params.count = this.params.count
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
      // x += this.left.data.get(i, j) * this.right.data.get(j, k)
      x += this.left.getData(i, j) * this.right.getData(j, k)
    }
    if (isNaN(x)) {
      console.log(`HEY dotprod_val(${i}, ${k}, ${minj}, ${maxj}) is NaN`)
    }
    const epi = this.params.epilog
    return epi == 'x/J' ? x / this.D :
      epi == 'x/sqrt(J)' || epi == 'softmax(x/sqrt(J))' ? x / Math.sqrt(this.D) :
        epi == 'tanh(x)' ? Math.tanh(x) :
          epi == 'relu(x)' ? Math.max(0, x) :
            x
  }

  getData(i, j) {
    return this.result.getData(i, j)
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

    this.group.clear()
    this.flow_guide_group = undefined

    // this.left.params = {'left/right': 'right/left', }

    if (this.params['arg orientation'] == 'positive/negative') {
      this.left.params['arg orientation'] = 'negative/positive'
      this.right.params['arg orientation'] = 'negative/positive'
    } else if (this.params['arg orientation'] == 'negative/positive') {
      this.left.params['arg orientation'] = 'positive/negative'
      this.right.params['arg orientation'] = 'positive/negative'
    }

    if (this.params['left placement'] == 'left/right') {
      this.left.params['left placement'] = 'right/left'
      this.right.params['left placement'] = 'right/left'
    } else if (this.params['left placement'] == 'right/left') {
      this.left.params['left placement'] = 'left/right'
      this.right.params['left placement'] = 'left/right'
    }

    if (this.params['right placement'] == 'top/bottom') {
      this.left.params['right placement'] = 'bottom/top'
      this.right.params['right placement'] = 'bottom/top'
    } else if (this.params['right placement'] == 'bottom/top') {
      this.left.params['right placement'] = 'top/bottom'
      this.right.params['right placement'] = 'top/bottom'
    }

    if (this.params['result placement'] == 'front/back') {
      this.left.params['result placement'] = 'back/front'
      this.right.params['result placement'] = 'back/front'
    } else if (this.params['result placement'] == 'back/front') {
      this.left.params['result placement'] = 'front/back'
      this.right.params['result placement'] = 'front/back'
    }

    this.initLeftVis()
    this.initRightVis()
    this.initResultVis()

    this.setFlowGuide()
    this.initAnimation()
    this.setRowGuides()
  }

  initLeftVis() {
    if (this.left) {
      this.group.remove(this.left.group)
    }
    this.left.initVis()

    if (this.params['arg orientation'].startsWith('positive')) {
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

    // TODO push down
    this.setLeftLegends()
  }

  initRightVis() {
    if (this.right) {
      this.group.remove(this.right.group)
    }
    this.right.initVis()

    if (this.params['arg orientation'].startsWith('positive')) {
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

    // TODO push down 
    this.setRightLegends()
  }

  initResultVis() {
    if (this.result) {
      this.group.remove(this.result.group)
    }

    this.result.initVis()

    if (this.params['result placement'].startsWith('back')) {
      this.result.group.position.z = this.getExtent().z
    }

    this.group.add(this.result.group)

    // TODO push down
    this.setResultLegends()
  }

  getPlacementInfo() {
    return {
      left: this.params['left placement'].startsWith('left') ? 1 : -1,
      right: this.params['right placement'].startsWith('top') ? 1 : -1,
      result: this.params['result placement'].startsWith('front') ? 1 : -1,
      zip: this.params['arg orientation'].startsWith('positive') ? 1 : -1,
      gap: this.params.gap,
      left_scatter: this.getLeftScatter(),
      right_scatter: this.getRightScatter(),
    }
  }

  setFlowGuide(enabled) {
    enabled = util.syncProp(this.params, 'flow guides', enabled)
    if (enabled) {
      if (!this.flow_guide_group) {
        this.flow_guide_group = util.flowGuide(this.H, this.D, this.W, this.getPlacementInfo())
        this.group.add(this.flow_guide_group)
      }
    } else {
      if (this.flow_guide_group) {
        this.group.remove(this.flow_guide_group)
        this.flow_guide_group = undefined
      }
    }
    this.left.setFlowGuide(enabled)
    this.right.setFlowGuide(enabled)
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
    if (this.anim_alg != 'none') {
      if (this.params['hide inputs']) {
        if (!hide) {
          this.params['hide inputs'] = false
          this.left.show()
          this.right.show()
        }
      } else {
        if (hide) {
          this.params['hide inputs'] = true
          this.left.hide()
          this.right.hide()
        }
      }
    }
  }

  setRowGuides(enabled) {
    enabled = util.syncProp(this.params, 'row guides', enabled)
    this.left.setRowGuides(enabled)
    this.right.setRowGuides(enabled)
    this.result.setRowGuides(enabled)
  }

  setLeftLegends() {
    const props = {
      name: this.params['left name'] || "X",
      height: "i",
      width: "j",
      hleft: true,
      wtop: false,
      ...(this.params.left_legend || {})
    }
    this.left.setLegends(this.params.legends, props)
  }

  setRightLegends() {
    const props = {
      name: this.params['right name'] || "Y",
      height: "j",
      width: "k",
      hleft: false,
      wtop: true,
      ...(this.params.right_legend || {})
    }
    this.right.setLegends(this.params.legends, props)
  }

  setResultLegends() {
    const props = {
      name: this.params['result name'] || "XY",
      height: "i",
      width: "k",
      hleft: false,
      wtop: false,
      ...(this.params.result_legend || {})
    }
    this.result.setLegends(this.params.legends, props)
  }

  setLegends(params) {
    util.updateProps(this.params, params)
    this.setLeftLegends()
    this.setRightLegends()
    this.setResultLegends()
  }

  // animation

  initAnimation() {
    this.anim_alg = this.params.alg || 'none'
    this.anim_mats = []

    if (this.anim_alg != 'none') {
      if (this.params['hide inputs']) {
        this.left.hide()
        this.right.hide()
      }
      this.result.hide()

      if (this.anim_alg == 'dotprod (row major)') {
        this.initAnimVmprod(true)
      } else if (this.anim_alg == 'dotprod (col major)') {
        this.initAnimMvprod(true)
      } else if (this.anim_alg == 'axpy') {
        this.initAnimVvprod(true)
      } else if (this.anim_alg == 'mvprod') {
        this.initAnimMvprod(false)
      } else if (this.anim_alg == 'vmprod') {
        this.initAnimVmprod(false)
      } else if (this.anim_alg == 'vvprod') {
        this.initAnimVvprod(false)
      }
    }
  }

  getThreadInfo() {
    const ni = Math.min(this.params['i threads'], this.H)
    const nj = Math.min(this.params['j threads'], this.D)
    const nk = Math.min(this.params['k threads'], this.W)
    return {
      i: { n: ni, block: Math.ceil(this.H / ni), max: this.H },
      j: { n: nj, block: Math.ceil(this.D / nj), max: this.D },
      k: { n: nk, block: Math.ceil(this.W / nk), max: this.W },
    }
  }

  grid(dims, f) {
    const info = this.getThreadInfo()
    const infos = Array.from(dims).map(d => info[d])
    const loop = (args, infos, f) => infos.length == 0 ?
      f(...args) :
      [...Array(infos[0].n).keys()].map(index => {
        const { block, max } = infos[0]
        const start = index * block
        if (start < max) {  // dead final block when block * n - max > block
          const end = Math.min(start + block, max)
          const extent = end - start
          loop([...args, { index, start, end, extent }], infos.slice(1), f)
        }
      })
    loop([], infos, f)
  }

  getAnimMatParams() {
    return { ...this.getLeafParams(), stretch_absmax: true }
  }

  getAnimResultMats() {
    if (this.getThreadInfo().j.n == 1) {
      this.result.params.stretch_absmax = true
      return [this.result]
    }
    const results = []
    this.grid('j', ({ start: j, end: je }) => {
      const result_init = (i, k) => this.dotprod_val(i, k, j, je)
      const data = Array2D.fromInit(this.H, this.W, result_init)
      const result = new Mat(data, this.getAnimMatParams(), true)
      result.group.position.z = je - 1
      result.group.rotation.x = Math.PI
      result.hide()
      results.push(result)
      this.group.add(result.group)
      this.anim_mats.push(result)
    })
    return results
  }

  initAnimVmprod(sweep) {
    const results = this.getAnimResultMats()

    const vmps = {}
    this.grid('ijk', ({ start: i }, { start: j, extent: jx }, { start: k, extent: kx }) => {
      const vmpinit = (ji, ki) => this.ijkmul(i, j + ji, k + ki)
      const data = Array2D.fromInit(jx, sweep ? 1 : kx, vmpinit)
      const vmp = new Mat(data, this.getAnimMatParams(), true)
      util.updateProps(vmp.group.position, { x: k, y: -i, z: j })
      vmp.group.rotation.x = Math.PI / 2
      vmps[[i, j, k]] = vmp
      this.anim_mats.push(vmp)
      this.group.add(vmp.group)
    })

    const { i: { block: iblock }, k: { block: kblock } } = this.getThreadInfo()
    let curi = iblock - 1
    let curk = sweep ? kblock - 1 : 0

    this.bump = () => {
      // update indexes
      const [oldi, oldk] = [curi, curk]
      if (sweep) {
        curk = (curk + 1) % kblock
      }
      if (curk == 0) {
        curi = (curi + 1) % iblock
      }

      // update result mats
      if (curi == 0 && curk == 0) {
        results.forEach(r => r.hide())
      }
      this.grid('ik', ({ start: i, extent: ix }, { start: k, end: ke, extent: kx }) => {
        if (curi < ix && curk < kx) {
          results.forEach(r => r.show(i + curi, sweep ? k + curk : [k, ke]))
        }
      })

      // update input hilights
      if (!this.params['hide inputs']) {
        if (sweep) {
          this.grid('k', ({ start: k, extent: kx }) => {
            if (oldk < kx) {
              this.right.setColorsAndSizes(undefined, k + oldk)
            }
            if (curk < kx) {
              this.right.bumpColor(undefined, k + curk)
            }
          })
        }
        if (oldi != curi) {
          this.grid('i', ({ start: i, extent: ix }) => {
            if (oldi < ix) {
              this.left.setColorsAndSizes(i + oldi, undefined)
            }
            if (curi < ix) {
              this.left.bumpColor(i + curi, undefined)
            }
          })
        }
      }

      // update intermediates
      this.grid('ijk', ({ start: i, extent: ix }, { start: j }, { start: k, extent: kx }) => {
        const vmp = vmps[[i, j, k]]
        if (curi < ix && curk < kx) {
          util.updateProps(vmp.group.position, { x: k + curk, y: -i - curi })
          vmp.reinit((ji, ki) => this.ijkmul(i + curi, j + ji, k + curk + ki))
        }
      })

      // update labels
      this.updateLabels()
    }
  }

  initAnimMvprod(sweep) {
    const gap = this.params.gap
    const { y: exty, z: extz } = this.getExtent()
    const results = this.getAnimResultMats()

    const mvps = {}
    this.grid('ijk', ({ start: i, extent: ix }, { start: j, extent: jx }, { start: k }) => {
      const mvpinit = (ii, ji) => this.ijkmul(i + ii, j + ji, k)
      const data = Array2D.fromInit(sweep ? 1 : ix, jx, mvpinit)
      const mvp = new Mat(data, this.getAnimMatParams(), true)
      mvp.hide()
      util.updateProps(mvp.group.position, { x: gap + k, z: extz + j })
      util.updateProps(mvp.group.rotation, { y: Math.PI / 2 })
      mvps[[i, j, k]] = mvp
      this.anim_mats.push(mvp)
      this.group.add(mvp.group)
    })

    const { i: { block: iblock }, k: { block: kblock } } = this.getThreadInfo()
    let curi = sweep ? iblock - 1 : 0
    let curk = kblock - 1

    this.bump = () => {
      // update indexes
      const [oldi, oldk] = [curi, curk]
      if (sweep) {
        curi = (curi + 1) % iblock
      }
      if (curi == 0) {
        curk = (curk + 1) % kblock
      }

      // update result mats
      if (curi == 0 && curk == 0) {
        results.forEach(r => r.hide())
      }
      this.grid('ik', ({ start: i, end: ie, extent: ix }, { start: k, extent: kx }) => {
        if (curi < ix && curk < kx) {
          results.forEach(r => r.show(sweep ? i + curi : [i, ie], k + curk))
        }
      })

      // update input hilights
      if (!this.params['hide inputs']) {
        if (sweep) {
          this.grid('i', ({ start: i, extent: ix }) => {
            if (oldi < ix) {
              this.left.setColorsAndSizes(i + oldi, undefined)
            }
            if (curi < ix) {
              this.left.bumpColor(i + curi, undefined)
            }
          })
        }
        if (oldk != curk) {
          this.grid('k', ({ start: k, extent: kx }) => {
            if (oldk < kx) {
              this.right.setColorsAndSizes(undefined, k + oldk)
            }
            if (curk < kx) {
              this.right.bumpColor(undefined, k + curk)
            }
          })
        }
      }

      // update intermediates
      this.grid('ijk', ({ start: i, extent: ix }, { start: j }, { start: k, extent: kx }) => {
        const mvp = mvps[[i, j, k]]
        if (curi < ix && curk < kx) {
          util.updateProps(mvp.group.position, { x: gap + k + curk, y: i + curi })
          mvp.reinit((ii, ji) => this.ijkmul(i + curi + ii, j + ji, k + curk))
        }
      })

      // update labels
      this.updateLabels()
    }
  }

  initAnimVvprod(sweep) {
    const results = this.getAnimResultMats()

    const vvps = {}
    this.grid('ijk', ({ start: i, extent: ix }, { start: j }, { start: k, extent: kx }) => {
      const vvpinit = (ii, ki) => this.ijkmul(i + ii, j, k + ki)
      const data = Array2D.fromInit(ix, sweep ? 1 : kx, vvpinit)
      const vvp = new Mat(data, this.getAnimMatParams(), true)
      util.updateProps(vvp.group.position, { x: k, y: -i, z: j })
      vvp.group.rotation.x = Math.PI
      vvps[[i, j, k]] = vvp
      this.anim_mats.push(vvp)
      this.group.add(vvp.group)
    })

    const { j: { block: jblock }, k: { block: kblock } } = this.getThreadInfo()
    let curj = jblock - 1
    let curk = sweep ? kblock - 1 : 0

    this.bump = () => {
      // update indexes
      const [oldj, oldk] = [curj, curk]
      curj = (curj + 1) % jblock
      if (curj == 0 && sweep) {
        curk = (curk + 1) % kblock
      }

      // update result mats
      if (curj == 0 && curk == 0) {
        results.forEach(r => r.hide())
      }
      this.grid('jk', ({ start: j, extent: jx, index: ji }, { start: k, end: ke, extent: kx }) => {
        if (curj < jx && curk < kx) {
          const f = (ii, ki) => this.dotprod_val(ii, ki, j, j + curj + 1)
          results[ji].reinit(f, undefined, undefined, sweep ? k + curk : [k, ke])
        }
      })

      // update input highlights
      if (!this.params['hide inputs']) {
        if (sweep) {
          this.grid('jk', ({ start: j, extent: jx }, { start: k, extent: kx }) => {
            if (oldj < jx && oldk < kx) {
              this.right.setColorsAndSizes(j + oldj, k + oldk)
            }
            if (curj < jx && curk < kx) {
              this.right.bumpColor(j + curj, k + curk)
            }
          })
        } else {
          this.grid('j', ({ start: j, extent: jx }) => {
            if (oldj < jx) {
              this.right.setColorsAndSizes(j + oldj, undefined)
            }
            if (curj < jx) {
              this.right.bumpColor(j + curj, undefined)
            }
          })
        }
        this.grid('j', ({ start: j, extent: jx }) => {
          if (oldj < jx) {
            this.left.setColorsAndSizes(undefined, j + oldj)
          }
          if (curj < jx) {
            this.left.bumpColor(undefined, j + curj)
          }
        })
      }

      // update intermediates
      this.grid('ijk', ({ start: i }, { start: j, extent: jx }, { start: k, extent: kx }) => {
        const vvp = vvps[[i, j, k]]
        if (curj < jx && curk < kx) {
          util.updateProps(vvp.group.position, { x: k + curk, z: j + curj })
          vvp.reinit((ii, ki) => this.ijkmul(i + ii, j + curj, k + curk + ki))
        }
      })

      // update labels
      this.updateLabels()
    }
  }
}