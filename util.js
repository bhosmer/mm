import * as THREE from 'three'

// https://threejs.org/examples/#webgl_buffergeometry_rawshader
const MMGUIDE_MATERIAL = new THREE.RawShaderMaterial({
  vertexShader: `
  precision mediump float;
  precision mediump int;
  uniform mat4 modelViewMatrix; // optional
  uniform mat4 projectionMatrix; // optional
  attribute vec3 position;
  attribute vec4 color;
  varying vec3 vPosition;
  varying vec4 vColor;
  void main()	{
    vPosition = position;
    vColor = color;
    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
  }`,
  fragmentShader: `
  precision mediump float;
  precision mediump int;
  varying vec3 vPosition;
  varying vec4 vColor;
  void main()	{
    vec4 color = vec4( vColor );
    gl_FragColor = color;
  }`,
  side: THREE.DoubleSide,
  transparent: true
});

//
// reading/writing params
//

export function updateFromSearchParams(obj, searchParams, strict = false) {

  function err(msg) {
    if (strict) {
      throw Error(msg)
    } else {
      console.log(`Warning: ${msg}`)
    }
  }

  const flattened = flatten(obj)
  for (const [k, v] of searchParams.entries()) {
    if (k in flattened) {
      const t = typeof flattened[k]
      const x = castToType(v, t)
      if (x !== undefined) {
        flattened[k] = x
      } else {
        err(`don't know how to cast param '${k}' to type ${t}`)
      }
    } else {
      err(`unknown param '${k}'`)
    }
  }

  updateProps(obj, unflatten(flattened))
}

// we only know a limited set of value types for simplicity
function castToType(v, t) {
  switch (t) {
    case 'boolean':
      return v == 'true'
    case 'number':
      return Number(v)
    case 'string':
      return String(v)
    default:
      return undefined
  }
}

//
// things with lines
//

export function lineSeg(start, end, color) {
  const material = new THREE.LineBasicMaterial({ color })
  const geometry = new THREE.BufferGeometry().setFromPoints([start, end])
  return new THREE.Line(geometry, material)
}

// x y z axis lines from origin
export function axes() {
  const origin = new THREE.Vector3(0, 0, 0)
  const group = new THREE.Group()
  group.add(lineSeg(origin, new THREE.Vector3(128, 0, 0), new THREE.Color(1, 0, 0)))
  group.add(lineSeg(origin, new THREE.Vector3(0, 128, 0), new THREE.Color(0, 1, 0)))
  group.add(lineSeg(origin, new THREE.Vector3(0, 0, 128), new THREE.Color(0, 0, 1)))
  return group
}

const CORNER_COLOR = new THREE.Uint8BufferAttribute([
  128, 128, 128, 64,
  128, 128, 128, 64,
  255, 255, 255, 128,
], 4)
CORNER_COLOR.normalized = true

export function rowGuide(h, w, light = 1.0) {
  const group = new THREE.Group()
  const color = new THREE.Color()

  const draw = (i0, j0, i1, j1) => {
    const start = new THREE.Vector3(j0, i0, 0)
    const end = new THREE.Vector3(j1, i1, 0)
    color.setHSL(1.0, 0.0, light)
    group.add(lineSeg(start, end, color))
  }

  draw(0, 0, h - 1, 0)
  draw(0, w - 1, h - 1, w - 1)

  const rstride = Math.max(1, (h - 1) / 8)
  for (let i = 0; i < h; i += rstride) {
    draw(i, 0, i, w - 1)
  }

  const corner_geometry = new THREE.BufferGeometry()
  corner_geometry.setAttribute('position', new THREE.Float32BufferAttribute([
    0, h / 8, 0,
    w / 8, 0, 0,
    0, 0, 0,
  ], 3))
  CORNER_COLOR.array[3] = CORNER_COLOR.array[7] = CORNER_COLOR.array[3] = 255 * light
  CORNER_COLOR.needsUpdate = true
  corner_geometry.setAttribute('color', CORNER_COLOR)
  group.add(new THREE.Mesh(corner_geometry, MMGUIDE_MATERIAL));

  return group
}

//
// mm flow guide arrow
// 

const ARROW_COLOR = new THREE.Uint8BufferAttribute([
  255, 128, 128, 255,
  128, 255, 128, 255,
  0, 128, 255, 255,
], 4)
ARROW_COLOR.normalized = true

export function flowGuide(h, d, w, placement, light = 1.0) {
  ARROW_COLOR.array[3] = ARROW_COLOR.array[7] = ARROW_COLOR.array[3] = 255 * light
  ARROW_COLOR.needsUpdate = true

  const { polarity, left, right, result, gap, left_scatter, right_scatter } = placement
  const extent = x => x + gap * 2 - 1
  const center = x => extent(x) / 2
  const place = (n, p, x) => p == 1 ? x : n - x
  const place_left = x => place(extent(w), left, x)
  const place_right = x => place(extent(h), right, x)
  const place_result = x => place(extent(d), result, x)

  const group = new THREE.Group()

  const left_geometry = new THREE.BufferGeometry()
  left_geometry.setAttribute('position', new THREE.Float32BufferAttribute([
    place_left(gap - left_scatter), center(h) - h / 8, place_result(center(d)),
    place_left(gap - left_scatter), center(h) + h / 8, place_result(center(d)),
    place_left(center(w)), place_right(center(h)), place_result(gap),
  ], 3))
  left_geometry.setAttribute('color', ARROW_COLOR)
  group.add(new THREE.Mesh(left_geometry, MMGUIDE_MATERIAL));

  const right_geometry = new THREE.BufferGeometry()
  right_geometry.setAttribute('position', new THREE.Float32BufferAttribute([
    center(w) - w / 8, place_right(gap - right_scatter), place_result(center(d)),
    center(w) + w / 8, place_right(gap - right_scatter), place_result(center(d)),
    center(w), place_right(center(h)), place_result(gap),
  ], 3))
  right_geometry.setAttribute('color', ARROW_COLOR)
  group.add(new THREE.Mesh(right_geometry, MMGUIDE_MATERIAL));

  return group
}

//
// bounding box stuff for text positioning
//

export function bbhw(g) {
  return {
    h: g.boundingBox.max.y - g.boundingBox.min.y,
    w: g.boundingBox.max.x - g.boundingBox.min.x
  }
}

export function center(x, y = 0) {
  return (x - y) / 2
}

//
// misc object utils
//

export function updateProps(obj, donor, f = (_, v) => v) {
  Object.entries(donor).map(([k, v]) => obj[k] = f(k, v))
}

export function syncProp(obj, k, v) {
  if (v === undefined) {
    return obj[k]
  }
  obj[k] = v
  return v
}

// {a: {b: 0, c: {d: 1}}} => {a$b: 0, a$c$d: 1}
// NOTE only handles our nested params - nothing null 
// or undefined, no arrays, no empty subobjects, etc
export function flatten(obj, sep = '.') {
  const f = (obj, pre) => Object.entries(obj).reduce((acc, [k, v]) => ({
    ...acc,
    ...(typeof v === 'object' ? f(obj[k], pre + k + sep) : { [pre + k]: v })
  }), {})
  return f(obj, '')
}

// {a$b: 0, a$c$d: 1} => {a: {b: 0, c: {d: 1}}}
export function unflatten(obj, sep = '.') {
  const add = (obj, [k, v]) => {
    const i = k.indexOf(sep)
    if (i >= 0) {
      const [base, suf] = [k.slice(0, i), k.slice(i + 1)]
      obj[base] = add(obj[base] || {}, [suf, v])
    } else {
      obj[k] = v
    }
    return obj
  }
  return Object.entries(obj).reduce(add, {})
}

// deepish - copies nested objects but not arrays
export function copyTree(obj) {
  return unflatten({ ...flatten(obj) })
}

// {my_prop_name: x} => {'my prop name': x}
export function spaces(obj) {
  return Object.entries(obj).reduce(
    (acc, [k, v]) => ({ ...acc, [k.replaceAll('_', ' ')]: v }),
    {}
  )
}

// {'my prop name': x} => {my_prop_name: x}
export function unders(obj) {
  return Object.entries(obj).reduce(
    (acc, [k, v]) => ({ ...acc, [k.replaceAll(' ', '_')]: v }),
    {}
  )
}

//
// misc THREE utils
//

export function disposeAndClear(obj) {
  obj.children && obj.children.map(disposeAndClear)
  obj.geometry && obj.geometry.dispose()
  obj.clear()
}