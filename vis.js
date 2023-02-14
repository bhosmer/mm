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

  absmin() {
    const data = this.data
    let absmin = Infinity
    for (let i = 0; i < data.length; i++) {
      const absx = Math.abs(data[i])
      if (absmin > absx) {
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
// Mat
//

const ELEM_SIZE = 1792

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
    this.h = data.h
    this.w = data.w
    this.absmax = this.data.absmax()
    this.absmin = this.data.absmin()

    this.points = emptyPoints(this.h, this.w)
    this.group = new THREE.Group()
    this.group.add(this.points)
    if (init_vis) {
      this.initVis()
    }
  }

  initVis(r = undefined, c = undefined) {
    const [rstart, rend] = toRange(r, this.h)
    const [cstart, cend] = toRange(c, this.w)
    const sizes = this.getPointSizes()
    const colors = this.getPointColors()
    for (let i = rstart; i < rend; i++) {
      for (let j = cstart, ptr = i * this.w + cstart; j < cend; j++, ptr++) {
        const x = this.data.data[ptr]
        sizes[ptr] = this.sizeFromData(x)
        this.setElemHSL(colors, ptr, x)
        this.checkLabel(ptr, x)
      }
    }
    this.points.geometry.attributes.pointSize.needsUpdate = true
    this.points.geometry.attributes.pointColor.needsUpdate = true
  }

  reinit(f, epi = undefined, r = undefined, c = undefined) {
    this.data.reinit(f, epi, r, c)
    if (this.params.stretch_limits) {
      this.absmin = Math.min(this.absmin, this.data.absmin())
      this.absmax = Math.max(this.absmax, this.data.absmax())
    }
    this.initVis(r, c)
  }

  getPointSizes() {
    return this.points.geometry.attributes.pointSize.array
  }

  getPointColors() {
    return this.points.geometry.attributes.pointColor.array
  }

  getGlobalAbsmax() {
    return this.params.getGlobalAbsmax ? this.params.getGlobalAbsmax() : this.absmax
  }

  sizeFromData(x) {
    if (x === undefined || isNaN(x)) {
      throw Error(`HEY sizeFromData(${x})`)
    }

    const local_sens = this.params.sensitivity == 'local'
    const absx = Math.abs(x)
    const [min, max] = local_sens ? [this.absmin, this.absmax] : [0, this.getGlobalAbsmax()]

    const vol = min == max ? 1 : (absx - min) / (max - min)
    const zsize = this.params['min size'] * ELEM_SIZE
    const size = zsize + (ELEM_SIZE - zsize) * Math.cbrt(vol)

    if (absx < min || absx > max || size < 0 || size > ELEM_SIZE || isNaN(size)) {
      throw Error(`HEY size ${size} absx ${absx} max ${max} min ${min} zsize ${zsize} sens ${local_sens}`)
    }

    return size
  }

  colorFromData(x) {
    if (x === undefined || isNaN(x)) {
      throw Error(`HEY colorFromData(${x})`)
    }

    const local_sens = this.params.sensitivity == 'local'
    const absx = Math.abs(x)
    const [min, max] = local_sens ? [this.absmin, this.absmax] : [0, this.getGlobalAbsmax()]

    const hue_vol = min == max ? x : x / (max - min)
    const gap = this.params['hue gap'] * Math.sign(x)
    const hue = (this.params['zero hue'] + gap + (Math.cbrt(hue_vol) * this.params['hue spread'])) % 1

    const light_vol = min == max ? 1 : (absx - min) / (max - min)
    const range = this.params['max light'] - this.params['min light']
    const light = this.params['min light'] + range * Math.cbrt(light_vol)

    return new THREE.Color().setHSL(hue, 1.0, light)
  }

  setElemHSL(a, i, x) {
    this.colorFromData(x).toArray(a, i * 3)
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

  getSize(i, j) {
    return this.points.geometry.attributes.pointSize.array[this.data.addr(i, j)]
  }

  setSize(i, j, x) {
    this.points.geometry.attributes.pointSize.array[this.data.addr(i, j)] = x
    this.points.geometry.attributes.pointSize.needsUpdate = true
  }

  setSizes(show, r, c) {
    const [rstart, rend] = toRange(r, this.h)
    const [cstart, cend] = toRange(c, this.w)
    const size = show ? (i, j) => this.sizeFromData(this.getData(i, j)) : (i, j) => 0
    for (let i = rstart; i < rend; i++) {
      for (let j = cstart; j < cend; j++) {
        this.setSize(i, j, size(i, j))
      }
    }
  }

  show(r = undefined, c = undefined) {
    this.setSizes(true, r, c)
  }

  hide(r = undefined, c = undefined) {
    this.setSizes(false, r, c)
  }

  isHidden(i, j) {
    return this.getSize(i, j) == 0 && this.sizeFromData(this.getData(i, j)) != 0
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

  setRowGuides(enabled = undefined) {
    enabled = util.syncProp(this.params, 'row guides', enabled)
    if (enabled) {
      if (!this.row_guide_group) {
        this.row_guide_group = util.rowGuide(this.h, this.w)
        this.group.add(this.row_guide_group)
      }
    } else {
      if (this.row_guide_group) {
        this.group.remove(this.row_guide_group)
        this.row_guide_group = undefined
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
      dim_size: sa_geo / 6,
    }
    const res = { ...defaults, ...custom }
    return res
  }

  setLegends(enabled, props) {
    if (enabled) {
      if (this.legends_group) {
        this.group.remove(this.legends_group)
        this.legends_group.clear()
      }
      this.legends_group = new THREE.Group()
      if (props.name) {
        const name = this.params.getText(props.name, props.name_color, props.name_size)
        const { h, w } = util.bbhw(name.geometry)
        name.geometry.rotateZ(Math.PI)
        name.geometry.rotateY(Math.PI)
        name.geometry.translate(util.center(this.w - 1, w), h + util.center(this.h - 1, h), -(1 + h / 2))
        this.legends_group.add(name)
      }
      if (props.height) {
        const height = this.params.getText(`${props.height} = ${this.h}`, props.dim_color, props.dim_size)
        const { h, w } = util.bbhw(height.geometry)
        height.geometry.rotateX(Math.PI)
        const zrot = (props.hleft ? -1 : 1) * Math.PI / 2
        height.geometry.rotateZ(zrot)
        const spacer = 0.5
        const xoff = props.hleft ? -h * 1 - spacer : this.w - 1 + h + spacer
        const yoff = props.hleft ? w + util.center(this.h - 1, w) : util.center(this.h - 1, w)
        height.geometry.translate(xoff, yoff, 0)
        this.legends_group.add(height)
      }
      if (props.width) {
        const width = this.params.getText(`${props.width} = ${this.w}`, props.dim_color, props.dim_size)
        const { h, w } = util.bbhw(width.geometry)
        width.geometry.rotateX(Math.PI)
        const spacer = 0.5
        const xoff = util.center(this.w - 1, w)
        const yoff = props.wtop ? -h * 1 - spacer : this.h - 1 + h * 1.5 + spacer
        width.geometry.translate(xoff, yoff, 0)
        this.legends_group.add(width)
      }
      this.group.add(this.legends_group)
    } else {
      if (this.legends_group) {
        this.group.remove(this.legends_group)
        this.legends_group.clear()
        this.legends_group = undefined
      }
    }
  }

  checkLabel(i, x) {
    if (this.label_cache) {
      const label = this.label_cache[i]
      if (label != undefined && label.value != x) {
        this.label_cache[i] = undefined
      }
    }
  }

  updateLabels(spotlight = undefined) {
    if (spotlight != undefined) {
      this.params.spotlight = spotlight
    } else {
      spotlight = this.params.spotlight
    }

    if (spotlight == 0) {
      if (this.label_group) {
        this.label_group.clear()
        this.group.remove(this.label_group)
        this.label_group = undefined
      }
      return
    }

    if (!this.label_group) {
      this.label_group = new THREE.Group()
      this.group.add(this.label_group)
      this.label_cache = []
    } else {
      this.label_group.clear()
    }

    this.params.raycaster.params.Points.threshold = spotlight
    this.params.raycaster.intersectObject(this.points).forEach(x => {
      const index = x.index
      const i = Math.floor(index / this.w)
      const j = index % this.w
      if (!this.isHidden(i, j)) {
        let label = this.label_cache[index]
        if (!label) {
          const x = this.getData(i, j)
          const fsiz = 0.175 - Math.log(1 + x / 10000)
          label = this.params.getText(`${x.toFixed(4)}`, 0xffffff, fsiz)
          label.value = x
          label.geometry.rotateX(Math.PI)
          const { h, w } = util.bbhw(label.geometry)
          label.geometry.translate(util.center(j * 2, w), h + util.center(i * 2, h), -0.25)
          this.label_cache[index] = label
        }
        this.label_group.add(label)
      }
    })
  }
}

//
// MatMul
//

export class MatMul {

  constructor(params) {
    this.params = { ...params }
    this.group = new THREE.Group()

    this.H = params.I
    this.D = params.J
    this.W = params.K

    this.initLeft()
    this.initRight()
    this.initResult()

    this.initVis()
  }

  getChildParams() {
    return { ...this.params, getGlobalAbsmax: this.getGlobalAbsmax.bind(this) }
  }

  initLeft() {
    if (this.left) {
      this.group.remove(this.left.group)
    }
    if (this.params.left) {
      this.left = this.params.left
      return
    }
    const data = this.params.left_data || (_ => {
      const init = this.params['left init']
      const min = this.params['left min']
      const max = this.params['left max']
      const sparsity = this.params['left sparsity']
      const f = getInitFunc(init, min, max, sparsity)
      return Array2D.fromInit(this.H, this.D, f)
    })()
    this.left = new Mat(data, this.getChildParams(), false)
  }

  initRight() {
    if (this.right) {
      this.group.remove(this.right.group)
    }
    if (this.params.right) {
      this.right = this.params.right
      return
    }
    const data = this.params.right_data || (_ => {
      const name = this.params['right init']
      const min = this.params['right min']
      const max = this.params['right max']
      const sparsity = this.params['right sparsity']
      const f = getInitFunc(name, min, max, sparsity)
      return Array2D.fromInit(this.D, this.W, f)
    })()
    this.right = new Mat(data, this.getChildParams(), false)
  }

  initResult() {
    if (this.result) {
      this.group.remove(this.result.group)
    }
    const result_init = (i, j) => this.dotprod_val(i, j)
    const data = Array2D.fromInit(this.H, this.W, result_init, this.params.epilog)
    this.result = new Mat(data, this.getChildParams(), false)
  }

  dotprod_val(i, k, minj = undefined, maxj = undefined) {
    if (minj === undefined) {
      minj = 0
    }
    if (maxj === undefined) {
      maxj = this.left.data.w
    }
    let x = 0.0
    for (let j = minj; j < maxj; j++) {
      x += this.left.data.get(i, j) * this.right.data.get(j, k)
    }
    if (isNaN(x)) {
      console.log(`HEY dotprod_val(${i}, ${j}, ${minj}, ${maxj}) is NaN`)
    }
    const epi = this.params.epilog
    return epi == 'x/J' ? x / this.D :
      epi == 'x/sqrt(J)' || epi == 'softmax(x/sqrt(J))' ? x / Math.sqrt(this.D) :
        epi == 'tanh(x)' ? Math.tanh(x) :
          epi == 'relu(x)' ? Math.max(0, x) :
            x
  }

  ijkmul(i, j, k) {
    return this.left.getData(i, j) * this.right.getData(j, k)
  }

  initVis(params = undefined) {
    if (params) {
      this.params = { ...params }
    }

    this.group.clear()
    this.flow_guide_group = undefined

    this.initLeftVis()
    this.initRightVis()
    this.initResultVis()

    this.setFlowGuide()
    this.setAnimation()
    this.setPosition()
    this.updateLabels()
  }

  initLeftVis() {
    if (this.params.left) {
      return
    }
    this.left.initVis()
    this.left.group.rotation.y = Math.PI / 2
    this.left.group.rotation.z = Math.PI
    if (this.params.left_rot) {
      Object.keys(this.params.left_rot).map(k => this.left.group.rotation[k] += this.params.left_rot[k])
    }
    this.left.group.position.x = -1
    if (this.params.left_pos) {
      Object.keys(this.params.left_pos).map(k => this.left.group.position[k] += this.params.left_pos[k])
    }
    this.left.setRowGuides()
    this.setLeftLegends(this.params.legends)

    this.group.add(this.left.group)
  }

  initRightVis() {
    if (this.params.right) {
      return
    }
    this.right.initVis()
    this.right.group.rotation.x = Math.PI / 2
    if (this.params.right_rot) {
      Object.keys(this.params.right_rot).map(k => this.right.group.rotation[k] += this.params.right_rot[k])
    }
    this.right.group.position.y = 1
    if (this.params.right_pos) {
      Object.keys(this.params.right_pos).map(k => this.right.group.position[k] += this.params.right_pos[k])
    }
    this.right.setRowGuides()
    this.setRightLegends(this.params.legends)
    this.group.add(this.right.group)
  }

  initResultVis() {
    this.result.initVis()
    this.result.group.rotation.x = Math.PI
    if (this.params.result_rot) {
      Object.keys(this.params.result_rot).map(k => this.result.group.rotation[k] += this.params.result_rot[k])
    }
    this.result.group.position.z = this.D
    if (this.params.result_pos) {
      Object.keys(this.params.result_pos).map(k => this.result.group.position[k] += this.params.result_pos[k])
    }
    this.result.setRowGuides()
    this.setResultLegends(this.params.legends)
    this.group.add(this.result.group)
  }

  updateLabels(spotlight = undefined) {
    this.left.updateLabels(spotlight)
    this.right.updateLabels(spotlight)
    this.result.updateLabels(spotlight)
    this.anim_mats.map(m => m.updateLabels(spotlight))
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

  getAbsmax() {
    return Math.max(this.left.absmax, this.right.absmax, this.result.absmax)
  }

  getGlobalAbsmax() {
    return this.params.getGlobalAbsmax ? this.params.getGlobalAbsmax() : this.getAbsmax()
  }

  setAnimation() {
    const prev_alg = this.anim_alg
    this.anim_alg = this.params.alg || 'none'
    if (this.anim_alg == prev_alg) {
      return
    }

    this.anim_mats = []
    if (prev_alg == 'vvprod' || prev_alg == 'axpy') {
      this.initResult() // clear depthwise accum into result
      this.result.params.stretch_limits = false
      this.initResultVis()
    }

    if (this.anim_alg == 'none') {
      this.left.show()
      this.right.show()
      this.result.show()
    } else {
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
      i: { n: ni, p: Math.floor(this.H / ni) },
      j: { n: nj, p: Math.floor(this.D / nj) },
      k: { n: nk, p: Math.floor(this.W / nk) },
    }
  }

  grid(dims, f) {
    const info = this.getThreadInfo()
    const lims = Array.from(dims).map(d => info[d].n)
    const loop = (ixs, lims, f) => lims.length == 0 ?
      f(...ixs) :
      [...Array(lims[0]).keys()].map(i => loop([...ixs, i], lims.slice(1), f))
    loop([], lims, f)
  }

  getAnimMatParams() {
    return { ...this.getChildParams(), stretch_limits: true }
  }

  getAnimResultMats() {
    const { j: { n: nj, p: jp } } = this.getThreadInfo()
    if (nj == 1) {
      this.result.params.stretch_limits = true
      return [this.result]
    }
    const results = []
    this.grid('j', j => {
      const result_init = (i, k) => this.dotprod_val(i, k, j * jp, (j + 1) * jp)
      const data = Array2D.fromInit(this.H, this.W, result_init)
      const result = new Mat(data, this.getAnimMatParams(), true)
      result.group.position.z = j * jp + jp - 1
      result.group.rotation.x = Math.PI
      result.hide()
      results.push(result)
      this.group.add(result.group)
      this.anim_mats.push(result)
    })
    return results
  }

  initAnimVmprod(sweep) {
    const { i: { p: ip }, j: { n: nj, p: jp }, k: { n: nk, p: kp } } = this.getThreadInfo()

    const results = this.getAnimResultMats()

    const vmps = []
    const vmpgroup = new THREE.Group()
    this.grid('ijk', (i, j, k) => {
      const vmpinit = (jx, kx) => this.ijkmul(i * ip, j * jp + jx, k * kp + kx)
      const data = Array2D.fromInit(jp, sweep ? 1 : kp, vmpinit)
      const vmp = new Mat(data, this.getAnimMatParams(), true)
      util.updateProps(vmp.group.position, { x: k * kp, y: -i * ip, z: j * jp })
      vmp.group.rotation.x = Math.PI / 2
      vmps.push(vmp)
      vmpgroup.add(vmp.group)
      this.anim_mats.push(vmp)
    })
    this.group.add(vmpgroup)

    let curi = ip - 1
    let curk = sweep ? kp - 1 : 0

    this.bump = () => {
      // update indexes
      const [oldi, oldk] = [curi, curk]
      if (sweep) {
        curk = (curk + 1) % kp
      }
      if (curk == 0) {
        curi = (curi + 1) % ip
      }

      // update result mats
      if (curi == 0 && curk == 0) {
        results.forEach(r => r.hide())
      }
      this.grid('ik', (i, k) =>
        results.forEach(r => r.show(i * ip + curi, sweep ? k * kp + curk : [k * kp, k * kp + kp]))
      )

      // update input hilights
      if (!this.params['hide inputs']) {
        if (oldi != curi) {
          this.grid('i', i => {
            this.left.bumpRowColor(i * ip + oldi, false)
            this.left.bumpRowColor(i * ip + curi, true)
          })
        }
        if (sweep) {
          this.grid('k', k => {
            this.right.bumpColumnColor(oldk + k * kp, false)
            this.right.bumpColumnColor(curk + k * kp, true)
          })
        }
      }

      // update intermediates
      util.updateProps(vmpgroup.position, { x: curk, y: -curi })
      this.grid('ijk', (i, j, k) => {
        const vmp = vmps[i * nj * nk + j * nk + k]
        vmp.reinit((jx, kx) => this.ijkmul(i * ip + curi, j * jp + jx, k * kp + kx + curk))
      })

      // update values
      this.updateLabels()
    }
  }

  initAnimMvprod(sweep) {
    const { i: { p: ip }, j: { n: nj, p: jp }, k: { n: nk, p: kp } } = this.getThreadInfo()

    const results = this.getAnimResultMats()

    const mvps = []
    const mvpgroup = new THREE.Group()
    this.grid('ijk', (i, j, k) => {
      const mvpinit = (ix, jx) => this.ijkmul(i * ip + ix, j * jp + jx, k * kp)
      const data = Array2D.fromInit(sweep ? 1 : ip, jp, mvpinit)
      const mvp = new Mat(data, this.getAnimMatParams(), true)
      util.updateProps(mvp.group.position, { x: k * kp, y: -i * ip, z: j * jp })
      util.updateProps(mvp.group.rotation, { y: Math.PI / 2, z: Math.PI })
      mvps.push(mvp)
      mvpgroup.add(mvp.group)
      this.anim_mats.push(mvp)
    })
    this.group.add(mvpgroup)

    let curi = sweep ? ip - 1 : 0
    let curk = kp - 1

    this.bump = () => {
      // update indexes
      const [oldi, oldk] = [curi, curk]
      if (sweep) {
        curi = (curi + 1) % ip
      }
      if (curi == 0) {
        curk = (curk + 1) % kp
      }

      // update result mats
      if (curi == 0 && curk == 0) {
        results.forEach(r => r.hide())
      }
      this.grid('ik', (i, k) =>
        results.forEach(r => r.show(sweep ? i * ip + curi : [i * ip, i * ip + ip], k * kp + curk))
      )

      // update input hilights
      if (!this.params['hide inputs']) {
        if (sweep) {
          this.grid('i', i => {
            this.left.bumpRowColor(i * ip + oldi, false)
            this.left.bumpRowColor(i * ip + curi, true)
          })
        }
        if (oldk != curk) {
          this.grid('k', k => {
            this.right.bumpColumnColor(k * kp + oldk, false)
            this.right.bumpColumnColor(k * kp + curk, true)
          })
        }
      }

      // update intermediates
      util.updateProps(mvpgroup.position, { x: curk, y: -curi })
      this.grid('ijk', (i, j, k) => {
        const mvp = mvps[i * nj * nk + j * nk + k]
        mvp.reinit((ix, jx) => this.ijkmul(i * ip + ix + curi, j * jp + jx, k * kp + curk))
      })

      // update values
      this.updateLabels()
    }
  }

  initAnimVvprod(sweep = false) {
    const { i: { p: ip }, j: { n: nj, p: jp }, k: { n: nk, p: kp } } = this.getThreadInfo()

    const results = this.getAnimResultMats()

    const vvps = []
    const vvpgroup = new THREE.Group()
    this.grid('ijk', (i, j, k) => {
      const vvpinit = (ix, kx) => this.ijkmul(i * ip + ix, j * jp, k * kp + kx)
      const data = Array2D.fromInit(ip, sweep ? 1 : kp, vvpinit)
      const vvp = new Mat(data, this.getAnimMatParams(), true)
      util.updateProps(vvp.group.position, { x: k * kp, y: -i * ip, z: j * jp })
      vvp.group.rotation.x = Math.PI
      vvps.push(vvp)
      vvpgroup.add(vvp.group)
      this.anim_mats.push(vvp)
    })
    this.group.add(vvpgroup)

    let curj = jp - 1
    let curk = sweep ? kp - 1 : 0

    this.bump = () => {
      // update indexes
      const [oldj, oldk] = [curj, curk]
      curj = (curj + 1) % jp
      if (curj == 0 && sweep) {
        curk = (curk + 1) % kp
      }

      // update result mats
      if (curj == 0 && curk == 0) {
        results.forEach(r => r.hide())
      }
      this.grid('jk', (j, k) => {
        const f = (i, kx) => this.dotprod_val(i, kx, j * jp, j * jp + curj + 1)
        results[j].reinit(f, undefined, undefined, sweep ? k * kp + curk : [k * kp, k * kp + kp])
      })

      // update input highlights
      if (!this.params['hide inputs']) {
        this.grid('j', j => {
          this.left.bumpColumnColor(j * jp + oldj, false)
          this.left.bumpColumnColor(j * jp + curj, true)
        })
        if (sweep) {
          this.grid('jk', (j, k) => {
            this.right.bumpColor(j * jp + oldj, k * kp + oldk, false)
            this.right.bumpColor(j * jp + curj, k * kp + curk, true)
          })
        } else {
          this.grid('j', j => {
            this.right.bumpRowColor(j * jp + oldj, false)
            this.right.bumpRowColor(j * jp + curj, true)
          })
        }
      }

      // update intermediates
      util.updateProps(vvpgroup.position, { x: curk, z: curj })
      this.grid('ijk', (i, j, k) => {
        const vvp = vvps[i * nj * nk + j * nk + k]
        vvp.reinit((ix, kx) => this.ijkmul(i * ip + ix, j * jp + curj, k * kp + kx + curk))
      })

      // update values
      this.updateLabels()
    }
  }

  setRowGuides(enabled) {
    enabled = util.syncProp(this.params, 'row guides', enabled)
    if (!this.params.left) {
      this.left.setRowGuides(enabled)
    }
    if (!this.params.right) {
      this.right.setRowGuides(enabled)
    }
    this.result.setRowGuides(enabled)
  }

  setFlowGuide(enabled) {
    enabled = util.syncProp(this.params, 'flow guides', enabled)
    if (enabled) {
      if (!this.flow_guide_group) {
        this.flow_guide_group = util.flowGuide(this.H, this.D, this.W)
        this.group.add(this.flow_guide_group)
      }
    } else {
      if (this.flow_guide_group) {
        this.group.remove(this.flow_guide_group)
        this.flow_guide_group = undefined
      }
    }
  }

  setLeftLegends(enabled) {
    const custom = this.params.left_legend ? this.params.left_legend : {}
    const defaults = { name: "X", height: "i", width: "j", hleft: true, wtop: false }
    const props = { ...this.left.getLegendProps(), ...defaults, ...custom }
    this.left.setLegends(enabled, props)
  }

  setRightLegends(enabled) {
    const custom = this.params.right_legend ? this.params.right_legend : {}
    const defaults = { name: "Y", height: "j", width: "k", hleft: false, wtop: true }
    const props = { ...this.right.getLegendProps(), ...defaults, ...custom }
    this.right.setLegends(enabled, props)
  }

  setResultLegends(enabled) {
    const custom = this.params.result_legend ? this.params.result_legend : {}
    const defaults = { name: "XY", height: "i", width: "k", hleft: false, wtop: false }
    const props = { ...this.result.getLegendProps(), ...defaults, ...custom }
    this.result.setLegends(enabled, props)
  }

  setNames(params) {
    let changed = false
    const left_name = params['left name']
    if (left_name != undefined && left_name != this.params['left name']) {
      this.params['left name'] = left_name
      this.params.left_legend = { ...(this.params.left_legend || {}), name: left_name }
      changed = true
    }
    const right_name = params['right name']
    if (right_name != undefined && right_name != this.params['right name']) {
      this.params['right name'] = right_name
      this.params.right_legend = { ...(this.params.right_legend || {}), name: right_name }
      changed = true
    }
    const result_name = params['result name']
    if (result_name != undefined && result_name != this.params['result name']) {
      this.params['result name'] = result_name
      this.params.result_legend = { ...(this.params.result_legend || {}), name: result_name }
      changed = true
    }
    if (changed) {
      this.setLegends(this.params.legends)
    }
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
}

//
// Attn (Q @ K.T) @ V
//

export class Attn {
  constructor(params) {
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

    this.initmm1()
    this.initmm2()
    // this.initAbsmax()

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
    this.mm1 = new MatMul(mm1_params)
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
    this.mm2 = new MatMul(mm2_params)
    this.group.add(this.mm2.group)
  }

  getGlobalAbsmax() {
    return Math.max(this.mm1.getAbsmax(), this.mm2.getAbsmax())
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
    this.mm1.initResult()
    this.initVis()
  }

  setResultEpilog(epilog) {
    this.params['result epilog'] = epilog
    this.mm2.initResult()
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
  constructor(params) {
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

        left_legend: { name: `L${i} `, height: `i${i} `, width: i == 0 ? "j0" : `j${i} = i${i - 1} ` },
        right_legend: i == 0 ? { name: `R${i} `, height: `j${i} `, width: "k" } : {},
        result_legend: { name: `R${i + 1} = L${i} R${i} `, height: `i${i} `, width: "k", hleft: i % 2 == 1 },

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

      const mm = new MatMul(mm_params)
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
  constructor(params) {
    this.params = { ...params }
    this.group = new THREE.Group()

    this.mms = []
    const nlayers = params.nlayers ? params.nlayers : 5
    for (let i = 0; i < nlayers; i++) {
      const I = params.I
      const J = i == 0 ? params.J_0 : params[`K_${i - 1} `]
      const K = params[`K_${i} `]
      const mm_params = {
        ...this.params,
        I: I,
        J: J,
        K: K,

        ...(i > 0 ? { left: this.mms[i - 1].result } : {}),

        left_legend: { name: `L${i} `, height: "i", width: `j${i} ` },
        right_legend: { name: `R${i} `, height: i == 0 ? "j0" : `j${i} = k${i - 1} `, width: `k${i} ` },
        result_legend: { name: `L${i + 1} = L${i} R${i} `, height: "i", width: `k${i} `, wtop: i % 2 == 1 },

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
      const mm = new MatMul(mm_params)
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
  constructor(params) {
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
    this.qmm = new MatMul(qmm_params)
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
    this.kmm = new MatMul(kmm_params)
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
    this.mm1 = new MatMul(mm1_params)
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
    this.vmm = new MatMul(vmm_params)
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
    this.mm2 = new MatMul(mm2_params)
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
    this.omm = new MatMul(omm_params)
    this.group.add(this.omm.group)
  }
}

export class Attn3 {
  constructor(params) {
    this.params = { ...params }
    this.group = new THREE.Group()

    this.H = params.n_q
    this.D = params.d_qk + params.d_v
    this.W = params.d_model

    this.initVis()
  }

  // moved from Mat - should go away in cleanup
  static matFromParams(h, w, params) {
    const init = initFuncFromParams(params.init)
    const data = params.data || Array2D.fromInit(h, w, init)
    const m = new Mat(data, params, false)

    m.setRowGuides()

    const custom = params.legend ? params.legend : {}
    const defaults = { name: "X", height: "i", width: "j", hleft: true, wtop: false }
    const props = { ...m.getLegendProps(), ...defaults, ...custom }
    m.setLegends(params.legends, props)

    return m
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
    const input = Attn3.matFromParams(this.params.n_q, this.params.d_model, input_params)
    input.group.rotation.x = Math.PI
    input.group.position.x = -this.params.d_model / 2
    input.group.position.y = this.params.n_q / 2
    input.group.position.z = -zspace
    this.group.add(input.group)
    this.params.input_data = input.data

    let output_data = input.data
    for (let i = 0; i < num_heads; i++) {
      const a = new Attn2(this.params)
      output_data = output_data.add(a.omm.result.data)
      if (show_heads) {
        a.group.position.z = i * (this.D + zspace)
        if (this.params['hide copied inputs']) {
          a.qmm.left.hide()
          a.kmm.right.hide()
          a.vmm.left.hide()
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
    const output = Attn3.matFromParams(this.params.n_q, this.params.d_model, output_params)
    output.group.rotation.x = Math.PI
    output.group.position.x = -this.params.d_model / 2
    output.group.position.y = this.params.n_q / 2
    output.group.position.z = (this.D + zspace) * num_heads
    this.group.add(output.group)

    //
    this.group.position.z = show_heads ? -(zspace * (num_heads - 1) + this.D * num_heads) / 2 : 0
  }
}
