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
    gl_PointSize = pointSize * 100.0 / -mvPosition.z;
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
// Array
//
class Array {
    constructor(h, w, f) {
        this.h = h
        this.w = w
        this.data = new Float32Array(h * w)
        this.absmax = 0
        let ptr = 0
        for (let i = 0; i < h; i++) {
            for (let j = 0; j < w; j++, ptr++) {
                const x = f(i, j, h, w)
                this.data[ptr] = x
                const absx = Math.abs(x)
                if (absx > this.absmax) {
                    this.absmax = absx
                }
            }
        }
    }

    get(i, j) {
        return this.data[this.addr(i, j)]
    }

    set(i, j, x) {
        const oldabsx = Math.abs(this.get(i, j))
        this.data[this.addr(i, j)] = x
        const absx = Math.abs(x)
        if (absx > this.absmax) {
            this.absmax = absx
        } else if (absx < this.absmax && oldabsx == this.absmax) {
            this.absmax = this.data.reduce(function (acc, x) {
                const absx = Math.abs(x)
                return absx > acc ? absx : acc
            })
        }
    }

    addr(i, j) {
        return i * this.w + j
    }
}

//
// Mat
//
class Mat {
    ELEM_SIZE = 20
    ELEM_SAT = 1.0
    ELEM_LIGHT = 0.6

    sizeFromData(x) {
        if (isNaN(x)) {
            return 0
        }
        const zsize = this.zero_size * this.ELEM_SIZE
        const range = (1 - this.zero_size) * this.ELEM_SIZE
        const size = zsize + range * Math.abs(x) / Math.max(this.container.global_absmax, 0.01)
        return size
    }

    setElemHSL(a, i, x, s = this.ELEM_SAT, l = this.ELEM_LIGHT) {
        if (isNaN(x)) {
            const c = new THREE.Color().setHSL(0, 0, 0)
            c.toArray(a, i * 3)
            return
        }

        const gap = (x == 0 ? 1 : Math.sign(x)) * this.hue_gap
        const h = (this.zero_hue + gap + (x / this.container.global_absmax * this.hue_spread)) % 1

        const lrange = this.max_light - this.zero_light
        l = this.zero_light + Math.abs(x) * lrange / Math.max(this.data.absmax, 0.01)

        const c = new THREE.Color().setHSL(h, s, l)
        c.toArray(a, i * 3)
    }

    static fromInit(h, w, init, container) {
        return new Mat(h, w, new Array(h, w, init), container)
    }

    constructor(h, w, data, container) {
        this.container = container
        this.zero_hue = container.params['zero hue']
        this.zero_size = container.params['zero size']
        this.zero_light = container.params['zero light']
        this.max_light = container.params['max light']
        this.hue_gap = container.params['hue gap']
        this.hue_spread = container.params['hue spread']
        this.h = h
        this.w = w
        this.data = data
        let sizes = new Float32Array(this.numel())
        let colors = new Float32Array(this.numel() * 3)
        let points = []
        for (let i = 0; i < this.h; i++) {
            for (let j = 0; j < this.w; j++) {
                points.push(new THREE.Vector3(j, i, 0))
                sizes[this.data.addr(i, j)] = this.sizeFromData(this.getData(i, j))
                this.setElemHSL(colors, this.data.addr(i, j), this.getData(i, j))
            }
        }
        const g = new THREE.BufferGeometry().setFromPoints(points);
        g.setAttribute('pointSize', new THREE.Float32BufferAttribute(sizes, 1))
        g.setAttribute('pointColor', new THREE.Float32BufferAttribute(colors, 3))
        this.points = new THREE.Points(g, MATERIAL)
    }

    numel() {
        return this.h * this.w
    }

    hideAll() {
        for (let i = 0; i < this.h; i++) {
            for (let j = 0; j < this.w; j++) {
                this.setSize(i, j, this.sizeFromData(NaN))
                this.setHSL(i, j, NaN)
            }
        }
    }

    showAll() {
        for (let i = 0; i < this.h; i++) {
            for (let j = 0; j < this.w; j++) {
                this.show(i, j)
            }
        }
    }

    show(i, j) {
        this.setSize(i, j, this.sizeFromData(this.getData(i, j)))
        this.setHSL(i, j, this.getData(i, j))
    }

    setSize(i, j, x) {
        this.points.geometry.attributes.pointSize.array[this.data.addr(i, j)] = x
        this.points.geometry.attributes.pointSize.needsUpdate = true
    }

    setHSL(i, j, h, s = this.ELEM_SAT, l = this.ELEM_LIGHT) {
        this.setElemHSL(this.points.geometry.attributes.pointColor.array, this.data.addr(i, j), h, s, l)
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

    setData(i, j, x) {
        this.data.set(i, j, x)
        this.setSize(i, j, this.sizeFromData(x))
        this.setHSL(i, j, x)
        this.container.setAbsmax()
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

//
// MatMul
//
export class MatMul {

    squeeze(x) {
        return this.init_base + this.init_range * x
    }

    getInitFunc(name, sparsity) {
        const gate = sparsity > 0 ?
            (f => Math.random() > sparsity ? f() : 0) :
            (f => f())
        switch (name) {
            case 'rows':
                return (i, j, h, w) => gate(() => this.squeeze(i / h))
            case 'cols':
                return (i, j, h, w) => gate(() => this.squeeze(j / w))
            case 'row major':
                return (i, j, h, w) => gate(() => this.squeeze((i * w + j) / (h * w)))
            case 'col major':
                return (i, j, h, w) => gate(() => this.squeeze((j * h + i) / (h * w)))
            case 'uniform':
                return (i, j, h, w) => gate(() => this.squeeze(Math.random()))
            case 'tril':
                return (i, j, h, w) => gate(() => (j <= i ? 1 : 0))
            case 'triu':
                return (i, j, h, w) => gate(() => (j >= i ? 1 : 0))
            case 'eye':
                return (i, j, h, w) => gate(() => (i == j ? 1 : 0))
            default:
                throw Error(`unrecognized initializer: ${name}`)
        }
    }

    constructor(params, getText, group = undefined) {
        this.getText = getText
        this.params = { ...params }

        this.group = new THREE.Group()
        if (group) {
            group.add(this.group)
        }

        this.H = params.I
        this.D = params.J
        this.W = params.K

        this.init_base = params['init min']
        this.init_range = Math.max(0, params['init max'] - params['init min'])

        this.initLeftData();
        this.initRightData();
        this.initResultData();

        this.initVis(params)
    }

    initLeftData() {
        if (this.params.left) {
            this.left_data = this.params.left.data
            return
        }
        const left_init = this.getInitFunc(this.params['left init'], this.params['left sparsity']);
        this.left_data = new Array(this.H, this.D, left_init);
    }

    initRightData() {
        if (this.params.right) {
            this.right_data = this.params.right.data
            return
        }
        const right_init = this.getInitFunc(this.params['right init'], this.params['right sparsity']);
        this.right_data = new Array(this.D, this.W, right_init);
    }

    initResultData() {
        const result_init = (y, x, h, w) => this._result_val(this.left_data, this.right_data, y, x);
        this.result_data = new Array(this.H, this.W, result_init);
    }

    initVis(params) {
        if (params) {
            this.params = { ...params }
        }
        this.group.clear()

        this._setAbsmax(this.left_data, this.right_data, this.result_data)

        this.initLeftVis();
        this.initRightVis();
        this.initResultVis();

        this.animation = 'none'
        this.setAnimation(this.params.animation)

        this.rowguides = []
        this.setGuides(this.params.guides)

        this.legends = []
        this.setLegends(this.params.legends)

        this.setPosition();
    }

    setPosition() {
        if (!this.params.pos) {
            // center cube on 0,0,0
            this.params.pos = new THREE.Vector3(-(this.W - 1) / 2, (this.H - 1) / 2, -(this.D - 1) / 2)
        }
        this.group.position.x = this.params.pos.x
        this.group.position.y = this.params.pos.y
        this.group.position.z = this.params.pos.z
    }

    initLeftVis() {
        if (this.params.left) {
            this.left = this.params.left
            return
        }
        if (this.left) {
            this.group.remove(this.left.points)
        }
        this.left = new Mat(this.H, this.D, this.left_data, this);
        this.left.points.rotation.y = Math.PI / 2;
        this.left.points.rotation.z = Math.PI;
        this.left.points.position.x = -1;
        this.group.add(this.left.points);
    }

    initRightVis() {
        if (this.params.right) {
            this.right = this.params.right
            return
        }
        if (this.right) {
            this.group.remove(this.right.points)
        }
        this.right = new Mat(this.D, this.W, this.right_data, this);
        this.right.points.rotation.x = Math.PI / 2;
        if (this.params.right_rot) {
            Object.keys(this.params.right_rot).map(k => this.right.points.rotation[k] += this.params.right_rot[k])
        }
        this.right.points.position.y = 1;
        if (this.params.right_pos) {
            Object.keys(this.params.right_pos).map(k => this.right.points.position[k] += this.params.right_pos[k])
        }
        this.group.add(this.right.points);
    }

    initResultVis() {
        if (this.result) {
            this.group.remove(this.result.points)
        }
        this.result = new Mat(this.H, this.W, this.result_data, this);
        this.result.points.rotation.x = Math.PI;
        if (this.params.result_rot) {
            Object.keys(this.params.result_rot).map(k => this.result.points.rotation[k] += this.params.result_rot[k])
        }
        this.result.points.position.z = this.D;
        if (this.params.result_pos) {
            Object.keys(this.params.result_pos).map(k => this.result.points.position[k] += this.params.result_pos[k])
        }
        this.group.add(this.result.points);
    }

    setEpilog(epilog) {
        this.params.epilog = epilog
        this.initResultData()
        this.initVis()
    }

    setI(i) {
        this.H = this.params.I = i
        this.initLeft()
    }

    initLeft(params = undefined) {
        if (params) {
            this.params = { ...params }
        }
        this.initLeftData();
        this.initResultData();
        this._setAbsmax(this.left_data, this.right_data, this.result_data);
        this.initVis(this.params);
    }

    setK(k) {
        this.W = this.params.K = k
        this.initRight();
    }

    initRight(params = undefined) {
        if (params) {
            this.params = { ...params }
        }
        this.initRightData();
        this.initResultData();
        this._setAbsmax(this.left_data, this.right_data, this.result_data);
        this.initVis(this.params);
    }

    setAnimation(animation) {
        this.animation = animation
        if (this.animation == 'dotprod') {
            this.result.hideAll()
            const dotprod_init = (y, x, h, w) => this.dotprod_val(0, 0, x)
            this.dotprod = Mat.fromInit(1, this.D, dotprod_init, this)
            this.dotprod.points.rotation.y = -Math.PI / 2
            this.group.add(this.dotprod.points)
            this.bump = this.bump_dotprod
            this.curi = this.H - 1
            this.curk = this.W - 1
        } else if (this.animation == 'itemwise') {
            this.result.hideAll()
            const itemwise_init = (y, x, h, w) => this.dotprod_val(0, 0, x)
            this.itemwise = Mat.fromInit(1, 1, itemwise_init, this)
            this.group.add(this.itemwise.points)
            this.bump = this.bump_itemwise
            this.curi = this.H - 1
            this.curj = this.D - 1
            this.curk = this.W - 1
        } else if (this.animation == 'mvprod') {
            this.result.hideAll()
            const mvprod_init = (y, x, h, w) => this.dotprod_val(0, y, x)
            this.mvprod = Mat.fromInit(this.H, this.D, mvprod_init, this)
            this.mvprod.points.rotation.y = Math.PI / 2
            this.mvprod.points.rotation.z = Math.PI
            this.group.add(this.mvprod.points)
            this.bump = this.bump_mvprod
            this.curk = this.W - 1
        } else if (this.animation == 'vmprod') {
            this.result.hideAll()
            const vmprod_init = (y, x, h, w) => this.dotprod_val(y, 0, x)
            this.vmprod = Mat.fromInit(this.D, this.W, vmprod_init, this)
            this.vmprod.points.rotation.x = Math.PI / 2
            this.group.add(this.vmprod.points)
            this.bump = this.bump_vmprod
            this.curi = this.H - 1
        } else if (this.animation == 'none') {
            this.result.showAll()
        }
    }

    dotprod_val(i, j, k) {
        return this.left.getData(i, k) * this.right.getData(k, j)
    }

    _result_val(a, b, i, j) {
        let x = 0.0
        const n = this.animation == 'itemwise' ? this.curk : a.w
        for (let k = 0; k < n; k++) {
            x += a.get(i, k) * b.get(k, j)
        }
        const epi = this.params.epilog
        return epi == 'x/J' ? x / this.D :
            epi == 'x/sqrt(J)' ? x / Math.sqrt(this.D) :
                epi == 'tanh(x)' ? Math.tanh(x) :
                    epi == 'relu(x)' ? Math.max(0, x) :
                        x
    }

    result_val(i, j) {
        return this._result_val(this.left.data, this.right.data, i, j)
    }

    _setAbsmax(a, b, c) {
        this.global_absmax = Math.max(a.absmax, b.absmax, c.absmax)
    }

    setAbsmax() {
        this._setAbsmax(this.left.data, this.right.data, this.result.data)
    }

    bump_vmprod() {
        const oldi = this.curi

        if (oldi < this.H - 1) {
            this.curi += 1
        } else {
            this.curi = 0
        }

        const i = this.curi

        // update result face
        if (this.curi == 0) {
            this.result.hideAll()
        }
        for (let k = 0; k < this.W; k++) {
            this.result.show(i, k)
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
        const oldk = this.curk

        if (oldk < this.W - 1) {
            this.curk += 1
        } else {
            this.curk = 0
        }

        const k = this.curk

        // update result face
        if (this.curk == 0) {
            this.result.hideAll()
        }
        for (let i = 0; i < this.H; i++) {
            this.result.show(i, k)
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
        const oldi = this.curi
        const oldk = this.curk

        if (oldk < this.W - 1) {
            this.curk += 1
        } else {
            this.curk = 0
            this.curi = oldi < this.H - 1 ? this.curi + 1 : 0
        }

        const i = this.curi
        const k = this.curk

        // update result face
        if (i == 0 && k == 0) {
            this.result.hideAll()
        }
        this.result.show(i, k)

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

    bump_itemwise() {
        const oldi = this.curi
        const oldj = this.curj
        const oldk = this.curk

        if (oldj < this.D - 1) {
            this.curj += 1
        } else {
            this.curj = 0
            if (oldk < this.W - 1) {
                this.curk += 1
            } else {
                this.curk = 0
                this.curi = oldi < this.H - 1 ? this.curi + 1 : 0
            }
        }

        const i = this.curi
        const j = this.curj
        const k = this.curk

        // update result face
        if (i == 0 && k == 0) {
            this.result.hideAll()
        }
        this.result.show(i, k)

        // hilight operand row/cols
        this.left.bumpColor(oldi, oldj, false)
        this.left.bumpColor(i, j, true)

        this.right.bumpColor(oldj, oldk, false)
        this.right.bumpColor(j, k, true)

        // move and recolor multiple item
        this.itemwise.points.position.x = this.right.points.geometry.attributes.position.array[k * 3]
        this.itemwise.points.position.y = -this.left.points.geometry.attributes.position.array[i * this.D * 3 + 1]
        this.itemwise.points.position.z = this.left.points.geometry.attributes.position.array[j * 3]
        this.itemwise.setData(0, 0, this.dotprod_val(i, k, j))
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

            this.rowguides.map(rg => this.group.add(rg))
        } else {
            this.rowguides.map(rg => this.group.remove(rg))
            this.rowguides = []
        }
    }

    // TODO devolve to Mat
    setLegends(enabled) {
        if (enabled) {
            const legend_color = 0x00aaff
            const legend_size = (this.H + this.D + this.W) / 75
            const name_color = 0xbbddff
            const name_size = (this.H + this.D + this.W) / 30

            const xname = this.getText("Q", name_color, name_size)
            const { h: xh, w: xw } = bbhw(xname.geometry)
            xname.geometry.rotateY(Math.PI / 2)
            xname.geometry.translate(-2, -xh - center(this.H - 1, xh), xw + center(this.D - 1, xw))
            this.legends.push(xname)

            const xhtext = this.getText("i = " + this.H, legend_color, legend_size)
            const { h: xhh, w: xhw } = bbhw(xhtext.geometry)
            xhtext.geometry.translate(center(this.H - 1, xhw), -2 * xhh, 1)
            xhtext.geometry.rotateX(-Math.PI / 2)
            xhtext.geometry.rotateY(-Math.PI)
            xhtext.geometry.rotateZ(Math.PI / 2)
            this.legends.push(xhtext)

            const xwtext = this.getText("j = " + this.D, legend_color, legend_size)
            const { h: xwh, w: xww } = bbhw(xwtext.geometry)
            xwtext.geometry.translate(center(this.D - 1, xww), -this.H - xwh, 1)
            xwtext.geometry.rotateY(-Math.PI / 2)
            this.legends.push(xwtext)

            const yname = this.getText("K^T", name_color, name_size)
            const { h: yh, w: yw } = bbhw(yname.geometry)
            yname.geometry.rotateX(-Math.PI / 2)
            yname.geometry.translate(center(this.W - 1, yw), 2, yh + center(this.D - 1, yh))
            this.legends.push(yname)

            const yhtext = this.getText("j = " + this.D, legend_color, legend_size)
            const { h: yhh, w: yhw } = bbhw(yhtext.geometry)
            yhtext.geometry.translate((this.D - 1) / 2 - yhw / 2, this.W + yhh / 2, 1)
            yhtext.geometry.rotateX(-Math.PI / 2)
            yhtext.geometry.rotateY(-Math.PI / 2)
            this.legends.push(yhtext)

            const ywtext = this.getText("k = " + this.W, legend_color, legend_size)
            const { h: ywh, w: yww } = bbhw(ywtext.geometry)
            ywtext.geometry.translate(center(this.W - 1, yww), ywh, 1)
            ywtext.geometry.rotateX(-Math.PI / 2)
            this.legends.push(ywtext)

            const zname = this.getText("A", name_color, name_size)
            const { h: zh, w: zw } = bbhw(zname.geometry)
            zname.geometry.translate(center(this.W - 1, zw), -zh - center(this.H - 1, zh), this.D + 1)
            this.legends.push(zname)

            const zhtext = this.getText("i = " + this.H, legend_color, legend_size)
            const { h: zhh, w: zhw } = bbhw(zhtext.geometry)
            zhtext.geometry.translate(center(this.H - 1, zhw), this.W + zhh / 2, this.D)
            zhtext.geometry.rotateZ(-Math.PI / 2)
            this.legends.push(zhtext)

            const zwtext = this.getText("k = " + this.W, legend_color, legend_size)
            const { h: zwh, w: zww } = bbhw(zwtext.geometry)
            zwtext.geometry.translate(center(this.W - 1, zww), -this.H - 1.5 * zwh, this.D)
            this.legends.push(zwtext)

            this.legends.map(leg => this.group.add(leg))
        } else {
            this.legends.map(leg => this.group.remove(leg))
            this.legends = []
        }
    }
}

//
// Attn
//

export class Attn {
    constructor(params, getText, group = undefined) {
        this.getText = getText
        this.group = group ? group : new THREE.Group()

        // TODO passed in
        const mm1_params = { ...params }
        const mm2_params = { ...params }

        if (mm1_params.I != mm2_params.I) {
            throw Error(`mm1_params.I ${mm1_params.I} mm2_params.I ${mm2_params.I}`)
        }
        if (mm1_params.K != mm2_params.J) {
            throw Error(`mm1_params.K ${mm1_params.K} mm2_params.J ${mm2_params.J}`)
        }

        this.H = mm1_params.I
        this.D = mm1_params.J + mm2_params.K
        this.W = mm1_params.K

        // TODO offset from parent pos
        this.mm1_params = mm1_params
        this.mm1_params.pos = new THREE.Vector3(-this.W / 2, this.H / 2, -this.D / 2)
        this.mm1 = new MatMul(this.mm1_params, getText, this.group)

        this.mm2_params = mm2_params
        this.mm2_params.left = this.mm1.result
        this.mm2_params.right_pos = new THREE.Vector3(0, -this.H - 1, mm2_params.K)
        this.mm2_params.right_rot = new THREE.Vector3(0, 0, -Math.PI / 2)
        this.mm2_params.result_pos = new THREE.Vector3(this.W, 0, 0)
        this.mm2_params.result_rot = new THREE.Vector3(0, -Math.PI / 2, 0)
        this.mm2_params.pos = new THREE.Vector3(-this.W / 2, this.H / 2, -this.D / 2 + mm1_params.J)
        this.mm2 = new MatMul(this.mm2_params, getText, this.group)
    }

    bump() {
        this.mm1.bump()
        this.mm2.bump()
    }

    // TODO devolve to Mat
    setGuides(enabled) {
        this.mm1.setGuides(enabled)
        this.mm2.setGuides(enabled)
    }

}


//
// misc
//

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

