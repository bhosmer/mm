/*
 1. harness.html for running simple examples
 2. grab some 3js examples and plug them into harness
 3. figure out how to make things show at all - materials, shaders, geometries etc.
 4. try swapping these into matmul
 5. other animations
 */


/*
1. bundle into class. TEST: two on screen at once
2. double matmul, chained matmul. TEST: attention
3. interaction. TEST: different sizes
4. different value functions, interactive
5. different interpretations (dot prod, left/right wsum)
6. execution patterns (parallel, block)
7. parallel slices
 */

import * as THREE from 'three'

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
    gl_PointSize = pointSize * 150.0 / -mvPosition.z;
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

const ELEM_SIZE = 16
const ELEM_SAT = 1.0
const ELEM_LIGHT = 0.6

function sizeFromData(x) {
  // return ELEM_SIZE * x
  return ELEM_SIZE
}

let BUMPS = []

function add_bump(f) {
  BUMPS.push(f)
}

function setHSL(a, i, h, s = ELEM_SAT, l = ELEM_LIGHT) {
  const c = new THREE.Color()
  c.setHSL(h, s, l)
  c.toArray(a, i * 3)
}

//
//
//
class Mat {

  addr(i, j) {
    return i * this.w + j
  }

  numel() {
    return this.h * this.w
  }

  constructor(h, w, f, xoff = 0, yoff = 0, zoff = 0) {
    this.h = h
    this.w = w
    this.data = new Float32Array(this.numel())
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        this.data[this.addr(i, j)] = f(i, j)
      }
    }
    this.initVis(xoff, yoff, zoff)
  }

  initVis(xoff, yoff, zoff) {
    let sizes = new Float32Array(this.numel())
    let colors = new Float32Array(this.numel() * 3)
    let points = []
    for (let i = 0; i < this.h; i++) {
      for (let j = 0; j < this.w; j++) {
        points.push(new THREE.Vector3(j + xoff, i + yoff, zoff))
        sizes[this.addr(i, j)] = sizeFromData(this.getData(i, j))
        setHSL(colors, this.addr(i, j), this.getData(i, j))
      }
    }
    const g = new THREE.BufferGeometry().setFromPoints(points);
    g.setAttribute('pointSize', new THREE.Float32BufferAttribute(sizes, 1))
    g.setAttribute('pointColor', new THREE.Float32BufferAttribute(colors, 3))
    this.points = new THREE.Points(g, MATERIAL)
  }

  getSize(i, j) {
    return this.points.geometry.attributes.pointSize.array[this.addr(i, j)]
  }

  setSize(i, j, x) {
    this.points.geometry.attributes.pointSize.array[this.addr(i, j)] = x
    this.points.geometry.attributes.pointSize.needsUpdate = true
  }

  getHSL(i, j) {
    const c = new THREE.Color()
    return c.fromArray(this.points.geometry.attributes.pointColor.array, this.addr(i, j) * 3).getHSL({})
  }

  setHSL(i, j, h, s = ELEM_SAT, l = ELEM_LIGHT) {
    setHSL(this.points.geometry.attributes.pointColor.array, this.addr(i, j), h, s, l)
    this.points.geometry.attributes.pointColor.needsUpdate = true
  }

  getColor(i, j) {
    const c = new THREE.Color()
    return c.fromArray(this.points.geometry.attributes.pointColor.array, this.addr(i, j) * 3)
  }

  setColor(i, j, c) {
    c.toArray(this.points.geometry.attributes.pointColor.array, this.addr(i, j) * 3)
    this.points.geometry.attributes.pointColor.needsUpdate = true
  }

  getData(i, j) {
    return this.data[this.addr(i, j)]
  }

  setData(i, j, x) {
    this.data[this.addr(i, j)] = x
    this.setSize(i, j, sizeFromData(x))
    this.setHSL(i, j, x)
  }

  fillData(x) {
    for (let i = 0; i < this.h; i++) {
      for (let j = 0; j < this.w; j++) {
        this.setData(i, j, 0)
      }
    }
  }

  faceLeft() {
    this.points.rotation.y = Math.PI / 2
    this.points.rotation.z = Math.PI
    return this
  }

  faceUp() {
    this.points.rotation.x = Math.PI / 2
    return this
  }

  faceFront() {
    this.points.rotation.x = Math.PI
    return this
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

}

// HEY

const H = 64 // 12 // 96 // 48
const W = 64 // 48 // 16 // 128 // 64
const D = 16 // 16 // 8 // 64 // 32

// 
//
//
function createMap1() {
  const map = new THREE.Group()

  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      // const k = Math.random() > 0.9 ? Math.floor(Math.random() * D) : 0
      const k = Math.random() > 0.9 ? 1 : 0
      // const col_init = (y, x) => squeeze((x * 10 / H) % 1)
      const col_init = (y, x) => squeeze(i / H)
      const col = new Mat(1, k, col_init).faceLeft()
      col.points.position.y = -i
      col.points.position.x = j
      map.add(col.points)
    }
  }

  // map.rotation.x = -Math.PI / 2
  // map.rotation.z = -Math.PI / 2

  map.position.x = 11 * W / 5 - W / 2
  map.position.y = H / 2


  return map
}



function createRagged1() {
  const ragged = new THREE.Group()

  const H = 64
  const W = 48
  const D = 8
  for (let i = 0; i < H; i++) {
    // const row_init = (y, x) => squeeze(x * i / H / 3)
    // const row_init = (y, x) => squeeze(i / H)
    // const row_init = (y, x) => Math.random()
    // const row_init = (y, x) => squeeze(i / H)
    const row_init = (y, x) => squeeze((i * 10 / H) % 1)

    // const row = new Mat(i, H - i, row_init).faceUp()
    // const w = (W / 4) + Math.floor(Math.random() * 3 * W / 4)
    // const w = Math.random() > 0.5 ? (Math.floor(Math.random() * W)) : 0
    const w = (W / 4) + Math.floor(Math.random() * 3 * W / 4)
    // const d = (D / 4) + Math.floor(Math.random() * 3 * D / 4)
    // const row = new Mat(d, w, row_init, -W / 2, 0, -D).faceUp()
    const row = new Mat(D, w, row_init, -W / 3, -D / 2, 0).faceUp()

    row.points.position.y = -i

    ragged.add(row.points)
  }

  // center cube on 0,0,0
  // ragged.position.x = -W / 2
  ragged.position.y = H / 2
  // ragged.position.z = D / 2

  return ragged
}

function createBlocks1() {
  const blocks = new THREE.Group()

  const H = 64
  const W = 48
  const D = 8

  let w, d
  let row_init

  let j = 0
  for (let i = 0; i < H; i++) {
    // if (i % 8 == 0) {
    j = j - 1
    if (j < 0) {
      j = 2 + Math.floor(Math.random() * 8)
      // row_init = (y, x) => squeeze(i / H)
      row_init = (y, x) => squeeze((i * 2 / H) % 1)
      // w = (W / 2) + Math.floor(Math.random() * W / 2)
      w = (W / 4) + Math.floor(Math.random() * 3 * W / 4)
      d = (D / 2) + Math.floor(Math.random() * D * 3 / 2)
    }

    // const row_init = (y, x) => squeeze(x * i / H / 3)
    // const row_init = (y, x) => squeeze(i / H)
    // const row_init = (y, x) => Math.random()

    const row = new Mat(d, w, row_init, -W / 3, -D / 2, 0).faceUp()

    row.points.position.y = -i
    blocks.add(row.points)
  }

  // center cube on 0,0,0
  // blocks.position.x = -1 * W / 5 - W / 2
  blocks.position.y = H / 2
  // ragged.position.z = -D / 2

  return blocks
}

function createTri1() {
  const tri = new THREE.Group()

  const H = 64
  const W = 32
  const D = 16
  for (let i = 0; i < H; i++) {
    // const row_init = (y, x) => squeeze(x * i / H / 3)
    // const row_init = (y, x) => squeeze(i / H)
    // const row_init = (y, x) => Math.random()
    const row_init = (y, x) => squeeze(i / H)

    // const h = H - i
    const h = i
    // const d = (D / 4) + Math.floor(Math.random() * 3 * D / 4)
    // const d = D
    const d = D
    const row = new Mat(h, d, row_init, 0, -W / 2)//.faceFront()
    row.points.position.z = -i

    // const row = new Mat(H - i, H - i, row_init).faceUp()
    // row.points.position.x = i
    // row.points.position.y = -i

    tri.add(row.points)
  }

  // center cube on 0,0,0
  // ragged.position.x = -W / 4
  // ragged.position.y = H / 2
  // ragged.position.z = -D / 2
  tri.rotation.x = -Math.PI / 2
  tri.rotation.z = -Math.PI / 2

  // tri.position.x = -7 * W / 5 - W / 2
  tri.position.y = H / 2
  // ragged.position.z = D - 1

  return tri
}


function createPy1() {
  const tri = new THREE.Group()

  const H = 64
  const W = 32
  const D = 8
  for (let i = 0; i < H; i++) {
    // const row_init = (y, x) => squeeze(x * i / H / 3)
    // const row_init = (y, x) => squeeze(i / H)
    // const row_init = (y, x) => Math.random()
    const row_init = (y, x) => squeeze(i / H)

    // const h = H - i
    const h = i
    // const d = (D / 4) + Math.floor(Math.random() * 3 * D / 4)
    // const d = D
    const d = i
    const row = new Mat(h, d, row_init, -W / 3, -D / 2, 0)//.faceFront()
    row.points.position.z = -i

    // const row = new Mat(H - i, H - i, row_init).faceUp()
    // row.points.position.x = i
    // row.points.position.y = -i

    tri.add(row.points)
  }

  // center cube on 0,0,0
  // ragged.position.x = -W / 4
  // ragged.position.y = H / 2
  // ragged.position.z = -D / 2
  tri.rotation.x = -Math.PI / 2
  tri.rotation.z = -Math.PI / 2

  // tri.position.x = -7 * W / 5 - W / 2
  tri.position.y = H / 2
  // ragged.position.z = D - 1

  return tri
}


function squeeze(x, base = .8, high = 1.8) {
  return base + (x * (high - base))
}

//
// animation
//

const PAUSE = 0
let LAST = 0

export function bump(t) {
  if (t - LAST < PAUSE) {
    return;
  }

  LAST = t

  BUMPS.forEach((f) => f())
}

// ---

export function createScene() {
  const scene = new THREE.Scene()

  // objects
  // const obj = createRagged1()
  // const obj = createBlocks1()
  const obj = createTri1()
  scene.add(obj)

  add_bump(() => {
    // obj.rotation.x += 0.01;
    // obj.rotation.y += 0.01;
    obj.rotation.z += 0.01;
  })

  // scene.add(createBlocks1())
  // scene.add(createTri1())
  // scene.add(createMap1())
  // scene.add(createDoubleMatMul())

  return scene
}


