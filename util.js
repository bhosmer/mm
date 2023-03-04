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

  for (const [k, v] of searchParams.entries()) {
    if (k in obj) {
      const t = typeof obj[k]
      const x = castToType(v, t)
      if (x !== undefined) {
        obj[k] = x
      } else {
        err(`don't know how to cast param '${k}' to type ${t}`)
      }
    } else {
      err(`unknown param '${k}'`)
    }
  }
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

function lineSeg(start, end, color) {
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

export function rowGuide(h, w, light = 1.0) {
  const rstride = Math.max(1, Math.floor(h / 8))
  const group = new THREE.Group()
  const color = new THREE.Color()

  const draw = (i0, j0, i1, j1, l) => {
    const start = new THREE.Vector3(j0, i0, 0)
    const end = new THREE.Vector3(j1, i1, 0)
    color.setHSL(1.0, 0.0, l)
    group.add(lineSeg(start, end, color))
  }

  for (let i = 0; i < h; i += rstride) {
    draw(i, 0, Math.min(i + rstride, h - 1), 0, light)
    draw(i, 0, i, w - 1, light)
  }

  return group
}

//
// mm flow guide arrow
// 

const ARROW_ATTR = new THREE.Uint8BufferAttribute([
  255, 128, 128, 224,
  128, 255, 128, 224,
  0, 128, 255, 224,
], 4)
ARROW_ATTR.normalized = true

export function flowGuide(h, d, w, placement, light = 1.0) {
  ARROW_ATTR.array[3] = ARROW_ATTR.array[7] = ARROW_ATTR.array[3] = 255 * light
  ARROW_ATTR.needsUpdate = true

  const { orientation, left, right, result, gap, left_scatter, right_scatter } = placement
  const extent = x => x + gap * 2 - 1
  const center = x => extent(x) / 2
  const place = (n, p, x) => p == 1 ? x : n - x
  const place_x = x => place(extent(w), left, x)
  const place_y = x => place(extent(h), right, x)
  const place_z = x => place(extent(d), result, x)

  const group = new THREE.Group()

  const left_geometry = new THREE.BufferGeometry()
  left_geometry.setAttribute('position', new THREE.Float32BufferAttribute([
    place_x(gap - left_scatter), place_y(center(h)), place_z(center(d) - (d / 6 * orientation)),
    place_x(gap - left_scatter), place_y(center(h)), place_z(center(d) + (d / 6 * orientation)),
    place_x(center(w)), place_y(center(h)), place_z(gap),
  ], 3))
  left_geometry.setAttribute('color', ARROW_ATTR)
  group.add(new THREE.Mesh(left_geometry, MMGUIDE_MATERIAL));

  const right_geometry = new THREE.BufferGeometry()
  right_geometry.setAttribute('position', new THREE.Float32BufferAttribute([
    place_x(center(w) - (w / 6 * orientation)), place_y(gap - right_scatter), place_z(center(d)),
    place_x(center(w) + (w / 6 * orientation)), place_y(gap - right_scatter), place_z(center(d)),
    place_x(center(w)), place_y(center(h)), place_z(gap),
  ], 3))
  right_geometry.setAttribute('color', ARROW_ATTR)
  group.add(new THREE.Mesh(right_geometry, MMGUIDE_MATERIAL));

  return group
}

//
// bounding box stuff for text positioning
//

export function bbhw(g) {
  return { h: g.boundingBox.max.y - g.boundingBox.min.y, w: g.boundingBox.max.x - g.boundingBox.min.x }
}

export function center(x, y = 0) {
  return (x - y) / 2
}

export function locate(y, x) {
  ['x', 'y', 'z'].map(d => y.rotation[d] = x.rotation[d]);
  ['x', 'y', 'z'].map(d => y.position[d] = x.position[d])
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