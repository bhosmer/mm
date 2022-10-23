// TODO
// make the hue/size actual matmul results
// ---
// hue fix
// size fix
// inners
// tf.js for training?
// h/v in examples
// randomized

import * as THREE from '../build/three.module.js'

const PARTICLE_MIN = 16
const PARTICLE_GROWTH = 36
const PARTICLE_MAX = PARTICLE_MIN + PARTICLE_GROWTH

const W = 64 // layer width (attn: sequence length K)
const H = 64 // batch size (attn: sequence length Q)
const D = 64 // example dims (attn: embedding dims)
const SCALE = 8
const GAP = SCALE * 2

const texture = new THREE.TextureLoader().load( 'textures/sprites/ball.png' );
// texture.wrapS = THREE.RepeatWrapping;
// texture.wrapT = THREE.RepeatWrapping;

const material = new THREE.ShaderMaterial({
  uniforms: {
    amplitude: { value: 1.0 },
    color: { value: new THREE.Color(0xffffff) },
    pointTexture: { value: texture }
  },
  vertexShader: document.getElementById('vertexshader').textContent,
  fragmentShader: document.getElementById('fragmentshader').textContent,
  // alphaTest: 0.9
})

let examples
let filters
let inners, innerg
let resultg

function result_hue (i, j = -1) {
  const x = i % W
  const y = Math.floor(i / W)

  const ec = examples.userData.colors
  const fc = filters.userData.colors

  let ex = y * D * 3
  let fx = x * D * 3

  const e = new THREE.Color()
  const f = new THREE.Color()

  // should only trigger for inners but
  // after init j == 0 on the face currently
  if (j >= 0) {
    ex += j * 3
    fx += j * 3
    const eh = e.fromArray(ec, ex).getHSL({}).h
    const fh = f.fromArray(fc, fx).getHSL({}).h
    // return Math.sqrt(eh * fh)
    return eh * fh
  }

  // j == -1, should be the face (result)
  let p = 0.0
  for (let i = 0; i < D; i++, ex += 3, fx += 3) {
    const eh = e.fromArray(ec, ex).getHSL({}).h
    const fh = f.fromArray(fc, fx).getHSL({}).h
    p += eh * fh
  }

  return p / D
}

function result_size (i, j = -1) {
  const x = i % W
  const y = Math.floor(i / W)

  const es = examples.userData.sizes
  const fs = filters.userData.sizes

  let ex = y * D
  let fx = x * D

  if (j >= 0) {
    // alert("HEY j >= 0 result_size")
    return PARTICLE_MAX - Math.abs(es[ex] - fs[fx])
  }

  let s = 0.0
  for (let i = 0; i < D; i++, ex++, fx++) {
    const e = es[ex]
    const f = fs[fx]
    // s += PARTICLE_MAX - Math.abs(f - e);
    s += (2 * (e * f)) / (PARTICLE_MAX * PARTICLE_MAX)
  }

  return PARTICLE_MIN + (s * PARTICLE_GROWTH) / D
}

function add (geom, h, s, l, z) {
  let vertices = geom.attributes.position.array

  let colors = new Float32Array(vertices.length)
  let sizes = new Float32Array(vertices.length / 3)

  const repeat = 1
  if (!h) h = i => (i * 3 * repeat) / vertices.length
  if (!s) s = i => 1.0
  if (!l) l = i => 0.5
  if (!z) z = i => PARTICLE_MIN + PARTICLE_GROWTH * ((i * 3) / vertices.length)

  // set color and size
  for (let i = 0, len = vertices.length / 3; i < len; i++) {
    let color = new THREE.Color()
    color.setHSL(h(i), s(i), l(i))
    color.toArray(colors, i * 3)
    sizes[i] = z(i)
  }

  let geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3))
  geometry.setAttribute('customColor', new THREE.BufferAttribute(colors, 3))
  geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1))

  // const material = new THREE.PointsMaterial( { size: 15, vertexColors: true } );

  let points = new THREE.Points(geometry, material)
  points.userData = { colors, sizes }

  return [points, geometry]
}

export function createScene () {
  let scene = new THREE.Scene()
  // scene.background = new THREE.Color( 0x050505 );
  // scene.fog = new THREE.Fog( 0xffffff, 2000, 3500 );

  const exg = new THREE.PlaneGeometry(D * SCALE, H * SCALE, D - 1, H - 1)
  let exampleg
  ;[examples, exampleg] = add(
    exg,
    undefined,
    undefined,
    undefined,
    i => (PARTICLE_MIN + PARTICLE_MAX) / 2
  )
  scene.add(examples)
  examples.rotation.y = -Math.PI / 2
  examples.position.x = -(W * (SCALE / 2) + GAP)

  const fig = new THREE.PlaneGeometry(D * SCALE, W * SCALE, D - 1, W - 1)
  // const filter_hue = i => ((i * W) + (i % W)) / fig.vertices.length;
  let filterg
  ;[filters, filterg] = add(
    fig,
    undefined,
    undefined,
    undefined,
    i => (PARTICLE_MIN + PARTICLE_MAX) / 2
  )
  scene.add(filters)
  filters.rotation.x = Math.PI / 2
  filters.position.y = H * (SCALE / 2) + GAP
  filters.position.z = -(W * (SCALE / 2) + GAP)
  examples.add(filters)

  // TODO LineGeometry
  const ing = new THREE.PlaneGeometry(D * SCALE, 1, D - 1, 1)
  // ;[inners, innerg] = add(ing, () => 0.5)
  ;[inners, innerg] = add(ing, result_hue, undefined, undefined, result_size)
  scene.add(inners)
  inners.rotation.x = Math.PI / 2
  inners.position.z = -GAP
  inners.position.y = H * (SCALE / 2)
  examples.add(inners)

  const resg = new THREE.PlaneGeometry(W * SCALE, H * SCALE, W - 1, H - 1)
  let results
  ;[results, resultg] = add(resg, result_hue, undefined, undefined, result_size)
  scene.add(results)
  results.rotation.y = Math.PI / 2
  results.position.x = D * (SCALE / 2) + GAP
  results.position.z = -(W * (SCALE / 2) + GAP)
  examples.add(results)

  return scene
}

let IJ = -1

export function bump () {

  const rpos = resultg.attributes.position.array
  const rsizes = resultg.attributes.size.array
  const n = rsizes.length

  const isizes = innerg.attributes.size.array
  const icolors = innerg.attributes.customColor.array
  const nc = icolors.length / 3

  const color = new THREE.Color()
  for (let k = 0; k < nc; k++) {
    color.setHSL(result_hue(IJ, k), 1.0, 0.6)
    color.toArray(icolors, k * 3)

    isizes[k] = result_size(IJ, k)
  }

  innerg.attributes.customColor.needsUpdate = true
  innerg.attributes.size.needsUpdate = true

  rsizes[IJ] = result_size(IJ)

  IJ = (IJ + 1) % n

  if (IJ == 0) {
    for (let i = 0; i < n; i++) {
      rsizes[i] = 0
    }
  }

  inners.position.y = rpos[IJ * 3 + 1]
  inners.position.z = -GAP - W * (SCALE / 2) - rpos[IJ * 3]

  resultg.attributes.size.needsUpdate = true
}


export function createCamera (window) {
  let camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 5, 3500 );
  camera.position.z = 2750;
  camera.position.x = -1000
  camera.position.y = 100
  camera.position.z = 2000
  return camera
}
