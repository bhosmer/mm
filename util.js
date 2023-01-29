
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

// draw line segment
function lineSeg(start, end, rgb, nsteps = 1000) {
  const points = []
  const colors = []
  const color = new THREE.Color(rgb)
  for (let i = 0; i < nsteps; i++) {
    points.push(start.x + (i * end.x / nsteps), start.y + (i * end.y / nsteps), (start.z + (i * end.z / nsteps)))
    colors.push(color.r, color.g, color.b)
  }

  const geometry = new LineGeometry()
  geometry.setPositions(points);
  geometry.setColors(colors);

  let material = new LineMaterial({
    linewidth: 1, // in world units with size attenuation, pixels otherwise
    vertexColors: true,
  })
  material.resolution.set(window.innerWidth, window.innerHeight) // resolution of the viewport

  let line = new THREE.Line(geometry, material);
  line.scale.set(2, 2, 2);
  return line
}

// make group with x y z axis lines
export function axes() {
  const origin = new THREE.Vector3(0, 0, 0)
  const group = new THREE.Group()
  group.add(lineSeg(origin, new THREE.Vector3(16, 0, 0), 0xff0000))
  group.add(lineSeg(origin, new THREE.Vector3(0, 16, 0), 0x00ff00))
  group.add(lineSeg(origin, new THREE.Vector3(0, 0, 16), 0x0000ff))
  return group
}

// make group with row guide lines
export function rowguide(nr, h, w, c = 0xffffff) {
  const group = new THREE.Group()
  const denom = 4
  if (w > 1) {
    const n = nr / denom
    const spacing = Math.max(Math.floor(n / 8), 1)
    for (let i = 0; i < n; i += spacing) {
      const start = new THREE.Vector3(0, i * h, 0)
      const end = new THREE.Vector3(w / denom * (1 - i / n), 0, 0)
      const ln = lineSeg(start, end, c, 2)
      group.add(ln)
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

