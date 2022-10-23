
import * as THREE from 'three'
import { LineGeometry } from 'https://threejs.org/examples/jsm/lines/LineGeometry.js';
import { LineMaterial } from 'https://threejs.org/examples/jsm/lines/LineMaterial.js';

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

// axes

export function axes(window) {
  const origin = new THREE.Vector3(0, 0, 0)
  const group = new THREE.Group()
  group.add(lineSeg(origin, new THREE.Vector3(16, 0, 0), 0xff0000))
  group.add(lineSeg(origin, new THREE.Vector3(0, 16, 0), 0x00ff00))
  group.add(lineSeg(origin, new THREE.Vector3(0, 0, 16), 0x0000ff))
  return group
}

// rowguide

export function rowguide(nr, h, w, c = 0xcccccc) {
  const group = new THREE.Group()
  const denom = 4
  if (w > 1) {
    let n = nr / denom
    for (let i = 0; i < n; i++) {
      const start = new THREE.Vector3(0, i * h, 0)
      const end = new THREE.Vector3(w / denom * (1 - i / n), 0, 0)
      const ln = lineSeg(start, end, c, 2)
      group.add(ln)
    }
  }
  return group
}

// ugh js

export function castToType(x, t, name) {
  switch (t) {
    case 'boolean':
      return x == 'true'
    case 'number':
      return Number(x)
    case 'string':
      return String(x)
    default:
      throw Error(`don't know how to cast to ${t}, x ${x} ${name ? name : ''}`)
  }
}