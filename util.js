
import * as THREE from 'three'
import { LineGeometry } from 'https://threejs.org/examples/jsm/lines/LineGeometry.js';
import { LineMaterial } from 'https://threejs.org/examples/jsm/lines/LineMaterial.js';

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

function lineSeg(start, end, color, width = 1) {
  const points = []
  const colors = []

  points.push(start.x, start.y, start.z)
  colors.push(color.r, color.g, color.b)
  points.push(end.x, end.y, end.z)
  colors.push(color.r, color.g, color.b)

  const geometry = new LineGeometry()
  geometry.setPositions(points)
  geometry.setColors(colors)

  const material = new LineMaterial({
    linewidth: width, // in world units with size attenuation, pixels otherwise
    vertexColors: true,
  })
  material.resolution.set(window.innerWidth, window.innerHeight) // resolution of the viewport

  return new THREE.Line(geometry, material)
}

// make group with x y z axis lines
export function axes() {
  const origin = new THREE.Vector3(0, 0, 0)
  const group = new THREE.Group()
  group.add(lineSeg(origin, new THREE.Vector3(32, 0, 0), new THREE.Color(1, 0, 0)))
  group.add(lineSeg(origin, new THREE.Vector3(0, 32, 0), new THREE.Color(0, 1, 0)))
  group.add(lineSeg(origin, new THREE.Vector3(0, 0, 32), new THREE.Color(0, 0, 1)))
  return group
}

// make group with row guide lines
export function rowguide(h, w) {
  const group = new THREE.Group()
  const color = new THREE.Color()
  const rstride = Math.max(1, Math.floor(h / 16))
  const cstride = Math.max(1, Math.floor(w / 16))
  const n = h * w

  const draw = (i0, j0, i1, j1) => {
    const start = new THREE.Vector3(j0, i0, 0);
    const end = new THREE.Vector3(j1, i1, 0);
    color.setScalar(((h - i0) * (w - j0)) / n)
    group.add(lineSeg(start, end, color))
  }

  for (let i = 0; i < h; i += rstride) {
    draw(i, 0, i + rstride, 0)
    for (let j = 0; j < w - 1; j += cstride) {
      draw(i, j, i, j + cstride)
    }
  }

  return group
}

//
// bounding box stuff for text positioning
//

export function bbhw(g) {
  return { h: g.boundingBox.max.y - g.boundingBox.min.y, w: g.boundingBox.max.x - g.boundingBox.min.x }
}

export function center(x, y) {
  return (x - y) / 2
}

export function locate(y, x) {
  ['x', 'y', 'z'].map(d => y.rotation[d] = x.rotation[d]);
  ['x', 'y', 'z'].map(d => y.position[d] = x.position[d])
}

//
// misc object utils
//

export function updateProps(obj, props) {
  Object.entries(props).map(([k, v]) => obj[k] = v)
}

