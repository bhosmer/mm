import * as THREE from 'three'
import * as util from './util.js'

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
    gl_PointSize = pointSize * 300.0 / -mvPosition.z;
    gl_Position = projectionMatrix * mvPosition;
  }
`,

  fragmentShader: `
  uniform vec3 color;
  uniform sampler2D pointTexture;
  varying vec4 vColor;

  void main() {
    vec4 outColor = texture2D( pointTexture, gl_PointCoord );
    // if ( outColor.a < 0.5 ) discard;
    gl_FragColor = outColor * vec4( color * vColor.xyz, 1.0 );
  }`,
})

function arrayFromInit(h, w, init) {
  const data = new Float32Array(h * w)
  let ptr = 0
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++, ptr++) {
      data[ptr] = init(i, j, h, w)
    }
  }
  return data
}

class Mat {
  ELEM_SIZE = 12
  ELEM_SIZE2 = .3
  ELEM_SAT = 1.0
  ELEM_LIGHT = 0.6

  sizeFromData(x) {
    return isNaN(x) ? 0 : (this.zero_size + x * (1 - this.zero_size)) * this.ELEM_SIZE
  }

  elemSize(i, j) {
    const x = this.getData(i, j)
    return isNaN(x) ? 0 : (this.zero_size + x * (1 - this.zero_size)) * this.ELEM_SIZE2
  }

  elemHue(i, j) {
    return this.zero_hue + this.getData(i, j)
  }

  setElemHSL(a, i, x, s = this.ELEM_SAT, l = this.ELEM_LIGHT) {
    const h = this.zero_hue + x
    const c = new THREE.Color()
    c.setHSL(h, s, l)
    c.toArray(a, i * 3)
  }

  static fromInit(h, w, init, params) {
    return new Mat(h, w, arrayFromInit(h, w, init), params)
  }

  constructor(h, w, data, params) {
    this.zero_hue = params['zero hue']
    this.zero_size = params['zero size']
    this.h = h
    this.w = w
    this.data = data

    this.group = new THREE.Group()
    for (let i = 0; i < this.h; i++) {
      for (let j = 0; j < this.w; j++) {
        // size
        const geometry = new THREE.SphereGeometry(this.elemSize(i, j), 16, 16);
        geometry.translate(j, i, 0)
        // color
        const material = new THREE.MeshStandardMaterial({
          color: new THREE.Color().setHSL(this.elemHue(i, j), this.ELEM_SAT, this.ELEM_LIGHT),
          side: THREE.DoubleSide,
        });
        this.group.add(new THREE.Mesh(geometry, material));

        // points.push(new THREE.Vector3(j, i, 0))
        // sizes[this.addr(i, j)] = this.sizeFromData(this.getData(i, j))
        // this.setElemHSL(colors, this.addr(i, j), this.getData(i, j))
      }
    }


    let sizes = new Float32Array(this.numel())
    let colors = new Float32Array(this.numel() * 3)
    let points = []
    for (let i = 0; i < this.h; i++) {
      for (let j = 0; j < this.w; j++) {
        // points.push(new THREE.Vector3(j, i, 0))
        points.push(new THREE.Vector3(0, 0, 0))
        sizes[this.addr(i, j)] = this.sizeFromData(this.getData(i, j))
        this.setElemHSL(colors, this.addr(i, j), this.getData(i, j))
      }
    }
    const g = new THREE.BufferGeometry().setFromPoints(points);
    g.setAttribute('pointSize', new THREE.Float32BufferAttribute(sizes, 1))
    g.setAttribute('pointColor', new THREE.Float32BufferAttribute(colors, 3))
    this.points = new THREE.Points(g, MATERIAL)
  }

  addr(i, j) {
    return i * this.w + j
  }

  numel() {
    return this.h * this.w
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

  setHSL(i, j, h, s = this.ELEM_SAT, l = this.ELEM_LIGHT) {
    this.setElemHSL(this.points.geometry.attributes.pointColor.array, this.addr(i, j), h, s, l)
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
    this.setSize(i, j, this.sizeFromData(x))
    this.setHSL(i, j, x)
  }

  fillData(x) {
    for (let i = 0; i < this.h; i++) {
      for (let j = 0; j < this.w; j++) {
        this.setData(i, j, x)
      }
    }
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

export class MatMul extends THREE.Group {

  squeeze(x) {
    return this.init_base + this.init_range * x
  }

  init_funcs = {
    'rows': (i, j, h, w) => this.squeeze((i * w + j) / (h * w)),
    'cols': (i, j, h, w) => this.squeeze((j * h + i) / (h * w)),
    'rand': (i, j, h, w) => this.squeeze(Math.random()),
    'sparse': (i, j, h, w) => Math.random() > 0.8 ? this.squeeze(Math.random()) : 0,
    'tril': (i, j, h, w) => (j <= i ? 1 : 0),
    'triu': (i, j, h, w) => (j >= i ? 1 : 0),
  }

  constructor(params, getText) {
    super()

    this.params = { ...params }

    this.H = params.I
    this.D = params.J
    this.W = params.K

    this.alg = params.alg
    this.result_scale = params['result scale']
    this.init_base = params['init min']
    this.init_range = params['init max'] - params['init min']

    this.left = Mat.fromInit(this.H, this.D, this.init_funcs[params['left init']], params)
    this.left.points.rotation.y = Math.PI / 2
    this.left.points.rotation.z = Math.PI
    this.left.points.position.x = -1
    this.add(this.left.points)
    this.left.group.rotation.y = Math.PI / 2
    this.left.group.rotation.z = Math.PI
    this.left.group.position.x = -1
    this.add(this.left.group)

    this.right = Mat.fromInit(this.D, this.W, this.init_funcs[params['right init']], params)
    this.right.points.rotation.x = Math.PI / 2
    this.right.points.position.y = 1
    this.add(this.right.points)
    this.right.group.rotation.x = Math.PI / 2
    this.right.group.position.y = 1
    this.add(this.right.group)

    const result_init = (y, x, h, w) => this.result_val(y, x)
    this.result = Mat.fromInit(this.H, this.W, result_init, params)
    this.result.points.rotation.x = Math.PI
    this.result.points.position.z = this.D
    this.result.fillData(NaN)
    this.add(this.result.points)
    this.result.group.rotation.x = Math.PI
    this.result.group.position.z = this.D
    this.add(this.result.group)

    if (this.alg == 'dotprod') {
      const dotprod_init = (y, x, h, w) => this.dotprod_val(0, 0, x)
      this.dotprod = Mat.fromInit(1, this.D, dotprod_init, params)
      this.dotprod.points.rotation.y = -Math.PI / 2
      this.add(this.dotprod.points)
      this.dotprod.group.rotation.y = -Math.PI / 2
      this.add(this.dotprod.group)
      this.bump = this.bump_dotprod
      this.I = this.H - 1
      this.K = this.W - 1
    } else if (this.alg == 'mvprod') {
      const mvprod_init = (y, x, h, w) => this.dotprod_val(0, y, x)
      this.mvprod = Mat.fromInit(this.H, this.D, mvprod_init, params)
      this.mvprod.points.rotation.y = Math.PI / 2
      this.mvprod.points.rotation.z = Math.PI
      this.add(this.mvprod.points)
      this.mvprod.group.rotation.y = Math.PI / 2
      this.mvprod.group.rotation.z = Math.PI
      this.add(this.mvprod.group)
      this.bump = this.bump_mvprod
      this.K = this.W - 1
    } else if (this.alg == 'vmprod') {
      const vmprod_init = (y, x, h, w) => this.dotprod_val(y, 0, x)
      this.vmprod = Mat.fromInit(this.D, this.W, vmprod_init, params)
      this.vmprod.points.rotation.x = Math.PI / 2
      this.add(this.vmprod.points)
      this.vmprod.group.rotation.x = Math.PI / 2
      this.add(this.vmprod.group)
      this.bump = this.bump_vmprod
      this.I = this.H - 1
    }

    this.rowguides = []
    this.setGuides(params['guides'])

    this.getText = getText
    this.legends = []
    this.setLegends(params.legends)

    // center cube on 0,0,0
    this.position.x = -(this.W - 1) / 2
    this.position.y = (this.H - 1) / 2
    this.position.z = -(this.D - 1) / 2
  }

  dotprod_val(y, x, z) {
    return this.left.getData(y, z) * this.right.getData(z, x)
  }

  result_val(y, x) {
    let p = 0.0
    const n = this.left.w
    for (let z = 0; z < n; z++) {
      p += this.left.getData(y, z) * this.right.getData(z, x)
    }
    return this.result_scale == 'J' ? p / this.D :
      this.result_scale == 'sqrt(J)' ? p / Math.sqrt(this.D) :
        this.result_scale == 'tanh' ? Math.tanh(p) :
          p
  }

  bump_vmprod() {
    const oldi = this.I

    if (oldi < this.H - 1) {
      this.I += 1
    } else {
      this.I = 0
    }

    const i = this.I

    // update result face
    if (this.I == 0) {
      this.result.fillData(NaN)
    }
    for (let k = 0; k < this.W; k++) {
      this.result.setData(i, k, this.result_val(i, k))
    }

    this.left.bumpRowColor(oldi, false)
    this.left.bumpRowColor(i, true)

    // move and recolor dot product vector
    this.vmprod.points.position.y = -this.left.points.geometry.attributes.position.array[i * this.D * 3 + 1]
    for (let x = 0; x < this.W; x++) {
      for (let z = 0; z < this.D; z++) {
        this.vmprod.setData(z, x, this.dotprod_val(i, x, z))
      }
    }
  }

  bump_mvprod() {
    const oldk = this.K

    if (oldk < this.W - 1) {
      this.K += 1
    } else {
      this.K = 0
    }

    const k = this.K

    // update result face
    if (this.K == 0) {
      this.result.fillData(NaN)
    }
    for (let i = 0; i < this.H; i++) {
      this.result.setData(i, k, this.result_val(i, k))
    }

    this.right.bumpColumnColor(oldk, false)
    this.right.bumpColumnColor(k, true)

    // move and recolor dot product vector
    this.mvprod.points.position.x = this.right.points.geometry.attributes.position.array[k * 3]
    for (let y = 0; y < this.H; y++) {
      for (let z = 0; z < this.D; z++) {
        this.mvprod.setData(y, z, this.dotprod_val(y, k, z))
      }
    }
  }

  bump_dotprod() {
    const oldi = this.I
    const oldk = this.K

    if (oldk < this.W - 1) {
      this.K += 1
    } else {
      this.K = 0
      this.I = oldi < this.H - 1 ? this.I + 1 : 0
    }

    const i = this.I
    const k = this.K

    // update result face
    if (i == 0 && k == 0) {
      this.result.fillData(NaN)
    }
    this.result.setData(i, k, this.result_val(i, k))

    // hilight operand row/cols
    if (oldk != k) {
      if (oldk >= 0) {
        this.right.bumpColumnColor(oldk, false)
      }
      this.right.bumpColumnColor(k, true)
    }

    if (oldi != i) {
      if (oldi >= 0) {
        this.left.bumpRowColor(oldi, false)
      }
      this.left.bumpRowColor(i, true)
    }

    // move and recolor dot product vector
    this.dotprod.points.position.x = this.right.points.geometry.attributes.position.array[k * 3]
    this.dotprod.points.position.y = -this.left.points.geometry.attributes.position.array[i * this.D * 3 + 1]
    for (let z = 0; z < this.dotprod.numel(); z++) {
      this.dotprod.setData(0, z, this.dotprod_val(i, k, z))
    }
  }

  // TODO devolve to Mat
  setGuides(enabled) {
    if (enabled) {
      const left_rowguide = util.rowguide(this.H, 0.5, this.D)
      locate(left_rowguide, this.left.points)
      this.rowguides.push(left_rowguide)

      const right_rowguide = util.rowguide(this.D, 0.5, this.W)
      locate(right_rowguide, this.right.points)
      this.rowguides.push(right_rowguide)

      const result_rowguide = util.rowguide(this.H, 0.5, this.W)
      locate(result_rowguide, this.result.points)
      this.rowguides.push(result_rowguide)

      this.rowguides.map(rg => this.add(rg))
    } else {
      this.rowguides.map(rg => this.remove(rg))
      this.rowguides = []
    }
  }

  // TODO devolve to Mat
  setLegends(enabled) {
    if (enabled) {
      const legend_color = 0x00aaff
      const legend_size = (this.H + this.D + this.W) ** 0.9 / 75
      const name_color = 0xbbddff
      const name_size = (this.H + this.D + this.W) ** 0.9 / 25

      const xname = this.getText("X", name_color, name_size)
      const { h: xh, w: xw } = bbhw(xname.geometry)
      xname.geometry.rotateY(Math.PI / 2)
      xname.geometry.translate(-2, -xh - center(this.H - 1, xh), xw + center(this.D - 1, xw))
      this.legends.push(xname)

      const xhtext = this.getText("I = " + this.H, legend_color, legend_size)
      const { h: xhh, w: xhw } = bbhw(xhtext.geometry)
      xhtext.geometry.translate(center(this.H - 1, xhw), -3 * xhh, 1)
      xhtext.geometry.rotateX(-Math.PI / 2)
      xhtext.geometry.rotateY(-Math.PI)
      xhtext.geometry.rotateZ(Math.PI / 2)
      this.legends.push(xhtext)

      const xwtext = this.getText("J = " + this.D, legend_color, legend_size)
      const { h: xwh, w: xww } = bbhw(xwtext.geometry)
      xwtext.geometry.translate(center(this.D - 1, xww), -this.H - xwh, 1)
      xwtext.geometry.rotateY(-Math.PI / 2)
      this.legends.push(xwtext)

      const yname = this.getText("Y", name_color, name_size)
      const { h: yh, w: yw } = bbhw(yname.geometry)
      yname.geometry.rotateX(-Math.PI / 2)
      yname.geometry.translate(center(this.W - 1, yw), 2, yh + center(this.D - 1, yh))
      this.legends.push(yname)

      const yhtext = this.getText("J = " + this.D, legend_color, legend_size)
      const { h: yhh, w: yhw } = bbhw(yhtext.geometry)
      yhtext.geometry.translate((this.D - 1) / 2 - yhw / 2, this.W + yhh, 1)
      yhtext.geometry.rotateX(-Math.PI / 2)
      yhtext.geometry.rotateY(-Math.PI / 2)
      this.legends.push(yhtext)

      const ywtext = this.getText("K = " + this.W, legend_color, legend_size)
      const { h: ywh, w: yww } = bbhw(ywtext.geometry)
      ywtext.geometry.translate(center(this.W - 1, yww), 3 * ywh, 1)
      ywtext.geometry.rotateX(-Math.PI / 2)
      this.legends.push(ywtext)

      const zname = this.getText("XY", name_color, name_size)
      const { h: zh, w: zw } = bbhw(zname.geometry)
      zname.geometry.translate(center(this.W - 1, zw), -zh - center(this.H - 1, zh), this.D + 1)
      this.legends.push(zname)

      const zhtext = this.getText("I = " + this.H, legend_color, legend_size)
      const { h: zhh, w: zhw } = bbhw(zhtext.geometry)
      zhtext.geometry.translate(center(this.H - 1, zhw), this.W, this.D)
      zhtext.geometry.rotateZ(-Math.PI / 2)
      this.legends.push(zhtext)

      const zwtext = this.getText("K = " + this.W, legend_color, legend_size)
      const { h: zwh, w: zww } = bbhw(zwtext.geometry)
      zwtext.geometry.translate(center(this.W - 1, zww), -this.H - zwh, this.D)
      this.legends.push(zwtext)

      this.legends.map(leg => this.add(leg))
    } else {
      this.legends.map(leg => this.remove(leg))
      this.legends = []
    }
  }
}

// --- 

function bbhw(geo) {
  return { h: bbh(geo), w: bbw(geo) }
}

function bbw(geo) {
  return geo.boundingBox.max.x - geo.boundingBox.min.x
}

function bbh(geo) {
  return geo.boundingBox.max.y - geo.boundingBox.min.y
}

function center(x, y) {
  return (x - y) / 2
}

function locate(y, x) {
  ['x', 'y', 'z'].map(d => y.rotation[d] = x.rotation[d]);
  ['x', 'y', 'z'].map(d => y.position[d] = x.position[d])
}


//# sourceURL=http://localhost:8000/vis.js