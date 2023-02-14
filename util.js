import * as THREE from 'three'

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

// row guide lines
export function rowGuide(h, w, trunc = true, rdenom = 16, cdenom = 16) {
  const rstride = trunc ? Math.max(1, Math.floor(h / rdenom)) : h / rdenom
  const cstride = trunc ? Math.max(1, Math.floor(w / cdenom)) : w / cdenom
  const n = h * w

  const group = new THREE.Group()
  const color = new THREE.Color()

  const draw = (i0, j0, i1, j1) => {
    const start = new THREE.Vector3(j0, i0, 0);
    const end = new THREE.Vector3(j1, i1, 0);
    color.setScalar(0.25 + 0.75 * (h - i0) * (w - j0) / n)
    group.add(lineSeg(start, end, color))
  }

  for (let i = 0; i < h; i += rstride) {
    draw(i, 0, Math.min(i + rstride, h - 1), 0)
    for (let j = 0; j < w; j += cstride) {
      draw(i, j, i, Math.min(j + cstride, w - 1))
    }
  }

  return group
}

//
// mm flow guide chevron
// 

// https://threejs.org/examples/#webgl_buffergeometry_rawshader
const MMGUIDE_MATERIAL = new THREE.RawShaderMaterial({
  uniforms: {
    time: { value: 1.0 }
  },

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

  uniform float time;

  varying vec3 vPosition;
  varying vec4 vColor;

  void main()	{

    vec4 color = vec4( vColor );
    color.r += sin( (vPosition.x + time) * 0.03 ) * 0.5;

    gl_FragColor = color;

  }`,

  side: THREE.DoubleSide,
  transparent: true
});

// TODO will need to be parameterized on orientation
export function flowGuide(h, d, w) {
  const colors = [
    179, 127, 199, 50,
    21, 138, 192, 163,
    134, 156, 249, 107,
  ]
  const color_attr = new THREE.Uint8BufferAttribute(colors, 4)
  color_attr.normalized = true

  const left_positions = [
    0.0, 0.0, 0.190983005625 * d,
    -0.5 * w, 0.0, 0.0,
    0.0, 0.0, 0.5 * d,
  ]
  const left_geometry = new THREE.BufferGeometry()
  left_geometry.setAttribute('position', new THREE.Float32BufferAttribute(left_positions, 3))
  left_geometry.setAttribute('color', color_attr)
  const left = new THREE.Mesh(left_geometry, MMGUIDE_MATERIAL)

  const right_positions = [
    0.0, 0.0, 0.190983005625 * d,
    0.0, 0.5 * h, 0.0,
    0.0, 0.0, 0.5 * d,
  ]
  const right_geometry = new THREE.BufferGeometry()
  right_geometry.setAttribute('position', new THREE.Float32BufferAttribute(right_positions, 3))
  right_geometry.setAttribute('color', color_attr)
  const right = new THREE.Mesh(right_geometry, MMGUIDE_MATERIAL)

  const group = new THREE.Group()
  group.add(left);
  group.add(right);
  group.position.x = center(w - 1)
  group.position.y = -center(h - 1)
  group.position.z = center(d - 1)
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

export function updateProps(obj, donor) {
  Object.entries(donor).map(([k, v]) => obj[k] = v)
}

export function syncProp(obj, k, v) {
  if (v == undefined) {
    return obj[k]
  }
  obj[k] = v
  return v
}