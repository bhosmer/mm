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

const ELEM_SIZE = 10
const ELEM_SAT = 1.0
const ELEM_LIGHT = 0.6

function sizeFromData(x) {
    return ELEM_SIZE * x
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

class Mat {

    addr(i, j) {
        return i * this.w + j
    }

    numel() {
        return this.h * this.w
    }

    constructor(h, w, f) {
        this.h = h
        this.w = w
        this.data = new Float32Array(this.numel())
        for (let i = 0; i < h; i++) {
            for (let j = 0; j < w; j++) {
                this.data[this.addr(i, j)] = f(i, j)
            }
        }
        this.initVis()
    }

    initVis() {
        let sizes = new Float32Array(this.numel())
        let colors = new Float32Array(this.numel() * 3)
        let points = []
        for (let i = 0; i < this.h; i++) {
            for (let j = 0; j < this.w; j++) {
                points.push(new THREE.Vector3(j, i, 0))
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

// const H = 32 // 16 // 12 // 96 // 48
// const D = 16 // 8 // 64 // 32
// const W = 16 // 48 // 16 // 128 // 64
// const D2 = 8


// const H = 32 // 12 // 96 // 48
// const D = 32 // 16 // 8 // 64 // 32
// const W = 4 // 48 // 16 // 128 // 64
// const D2 = 32

const H = 32 // 27 // 12 // 96 // 48
const D = 32 // 18 // 16 // 8 // 64 // 32
const W = 32 // 12 // 48 // 16 // 128 // 64
const D2 = 4 // 8


class MatMul {
    // @@@ move everything into here
}

let left
let right
let result
let dotprod

function dotprod_val(y, x, z) {
    return left.getData(y, z) * right.getData(z, x)
}

function result_val(y, x) {
    let p = 0.0
    for (let z = 0; z < left.w; z++) {
        p += left.getData(y, z) * right.getData(z, x)
    }
    return p / left.w
}

let IJ = 0
let K = 0

function bump_mm() {
    const oldi = Math.floor(IJ / W)
    const oldj = IJ % W

    IJ = (IJ + 1) % result.numel()

    const i = Math.floor(IJ / W)
    const j = IJ % W

    // update result face
    if (IJ == 0) {
        result.fillData(0)
    }
    result.setData(i, j, result_val(i, j))

    // hilight operand row/cols
    if (oldj != j) {
        if (oldj >= 0) {
            right.bumpColumnColor(oldj, false)
        }
        right.bumpColumnColor(j, true)
    }

    if (oldi != i) {
        if (oldi >= 0) {
            left.bumpRowColor(oldi, false)
        }
        left.bumpRowColor(i, true)
    }

    // move and recolor dot product vector
    dotprod.points.position.x = right.points.geometry.attributes.position.array[j * 3]
    dotprod.points.position.y = -left.points.geometry.attributes.position.array[Math.floor(IJ * D / W) * 3 + 1]
    for (let z = 0; z < dotprod.numel(); z++) {
        dotprod.setData(0, z, dotprod_val(i, j, z))
    }
}

export function createMatMul() {
    const mm = new THREE.Group()

    // const left_init = (y, x) => 0.5 + Math.abs(Math.sin(y) * 10 + Math.abs(Math.cos(x) * 10)) % 0.5
    const left_init = (y, x) => y / H
    left = new Mat(H, D, left_init).faceLeft()
    left.points.position.x = -1
    mm.add(left.points)

    // const right_init = (y, x) => 0.5 + Math.abs(Math.sin(x) * 10 + Math.abs(Math.cos(y) * 10)) % 0.5
    // const right_init = (y, x) => x / W
    const right_init = (y, x) => y / H
    right = new Mat(D, W, right_init).faceUp()
    right.points.position.y = 1
    mm.add(right.points)

    result = new Mat(H, W, result_val).faceFront()
    result.points.position.z = D
    mm.add(result.points)

    const dotprod_init = (y, x) => dotprod_val(0, 0, x)
    dotprod = new Mat(1, D, dotprod_init).faceLeft()
    mm.add(dotprod.points)

    // center cube on 0,0,0
    mm.position.x = -W / 2
    mm.position.y = H / 2
    mm.position.z = -D / 2

    add_bump(bump_mm)

    return mm
}

let left2
let right2
let result2
let dotprod2

function result2_attn_val(y, x) {
    let p = 0.0
    for (let z = 0; z < result.w; z++) {
        p += result.getData(y, z) * right2.getData(z, x)
    }
    return p / result.w
}

function dotprod2_attn_val(y, x, z) {
    return result.getData(y, z) * right2.getData(z, x)
}

function result2_mlp_val(y, x) {
    let p = 0.0
    for (let z = 0; z < left2.h; z++) {
        p += result.getData(z, y) * left2.getData(z, x)
    }
    return p / left2.h
}

function dotprod2_mlp_val(y, x, z) {
    return result.getData(y, z) * left2.getData(z, x)
}

let PHASE = 1

function bump_attn_mm2() {
    let i, j, oldi, oldj

    if (PHASE == 0) {
        oldi = Math.floor(IJ / W)
        oldj = IJ % W

        // HEY
        IJ = (IJ + 1) % result.numel()
        // IJ += W
        // if (IJ >= result.numel()) {
        //   IJ -= result.numel() - 1
        //   if (IJ == W) {
        //     IJ = 0
        //   }
        // }

        i = Math.floor(IJ / W)
        j = IJ % W
    } else {
        oldi = Math.floor(IJ / D2)
        oldj = IJ % D2

        IJ = (IJ + 1) % result2.numel()
        // HEY
        // IJ += D2
        // if (IJ >= result2.numel()) {
        //   IJ -= result2.numel() - 1
        //   if (IJ == D2) {
        //     IJ = 0
        //   }
        // }

        i = Math.floor(IJ / D2)
        j = IJ % D2
    }

    if (IJ == 0) {
        if (PHASE == 0) {
            dotprod.fillData(0)
            left.bumpRowColor(oldi, false)
            right.bumpColumnColor(oldj, false)
        } else {
            result.fillData(0)
            result2.fillData(0)
            result.bumpRowColor(oldi, false)
            right2.bumpColumnColor(oldj, false)
            dotprod2.fillData(0)
        }
        PHASE = (PHASE + 1) % 2
    }

    // update result face
    if (PHASE == 0) {
        result.setData(i, j, result_val(i, j))

        // hilight operand row/cols
        if (oldj != j) {
            if (oldj >= 0) {
                right.bumpColumnColor(oldj, false)
            }
            right.bumpColumnColor(j, true)
        }

        if (oldi != i) {
            if (oldi >= 0) {
                left.bumpRowColor(oldi, false)
            }
            left.bumpRowColor(i, true)
        }

        // move and recolor dot product vector
        dotprod.points.position.x = right.points.geometry.attributes.position.array[j * 3]
        dotprod.points.position.y = -left.points.geometry.attributes.position.array[Math.floor(IJ * D / W) * 3 + 1]
        for (let z = 0; z < dotprod.numel(); z++) {
            dotprod.setData(0, z, dotprod_val(i, j, z))
        }
    } else {
        result2.setData(i, j, result2_attn_val(i, j))

        // hilight operand row/cols
        if (oldj != j) {
            if (oldj >= 0) {
                right2.bumpColumnColor(oldj, false)
            }
            right2.bumpColumnColor(j, true)
        }

        if (oldi != i) {
            if (oldi >= 0) {
                result.bumpRowColor(oldi, false)
            }
            result.bumpRowColor(i, true)
        }

        // move and recolor dot product vector
        dotprod2.points.position.z = right2.points.geometry.attributes.position.array[j * 3] + D + 1
        dotprod2.points.position.y = -result.points.geometry.attributes.position.array[Math.floor(IJ * W / D2) * 3 + 1]
        for (let z = 0; z < dotprod2.numel(); z++) {
            dotprod2.setData(0, z, dotprod2_attn_val(i, j, z))
        }
    }
}

function squeeze(x, base = .5, high = 1.4) {
    return base + (x * (high - base))
}

export function createAttnDoubleMatMul() {
    const mm2 = new THREE.Group()

    // HEY
    // const left_init = (y, x) => squeeze(y / H)
    const left_init = (y, x) => squeeze(x / D)
    // const left_init = (y, x) => Math.random()
    left = new Mat(H, D, left_init).faceLeft()
    left.points.position.x = -1
    mm2.add(left.points)

    // HEY
    // const right_init = (y, x) => squeeze(y / D)
    const right_init = (y, x) => squeeze(x / W)
    // const right_init = (y, x) => Math.random()
    right = new Mat(D, W, right_init).faceUp()
    right.points.position.y = 1
    mm2.add(right.points)

    result = new Mat(H, W, result_val).faceFront()
    result.points.position.z = D
    result.fillData(0)
    mm2.add(result.points)

    const dotprod_init = (y, x) => dotprod_val(0, 0, x)
    dotprod = new Mat(1, D, dotprod_init).faceLeft()
    mm2.add(dotprod.points)

    // HEY
    // const right2_init = (y, x) => squeeze(y / W)
    const right2_init = (y, x) => squeeze(x / D2)
    // const right2_init = (y, x) => Math.random()
    right2 = new Mat(W, D2, right2_init)
    right2.points.rotation.x = -Math.PI / 2
    right2.points.rotation.z = -Math.PI / 2
    right2.points.position.y = -H
    right2.points.position.z = D + 1
    mm2.add(right2.points)

    result2 = new Mat(H, D2, result2_attn_val).faceLeft() // TODO right
    result2.points.position.x = W
    result2.points.position.z = D + 1
    result2.fillData(0)
    mm2.add(result2.points)

    const dotprod2_init = (y, x) => dotprod2_attn_val(0, 0, x)
    dotprod2 = new Mat(1, W, dotprod2_init)
    mm2.add(dotprod2.points)

    // center cube on 0,0,0
    mm2.position.x = -W / 2
    mm2.position.y = H / 2
    mm2.position.z = -(D + D2) / 2

    // HEY
    IJ = result.numel() - 1

    add_bump(bump_attn_mm2)

    return mm2
}

function bump_mlp_mm2() {
    let i, j, oldi, oldj

    if (PHASE == 0) {
        oldi = Math.floor(IJ / W)
        oldj = IJ % W

        // HEY
        IJ = (IJ + 1) % result.numel()
        // IJ += W
        // if (IJ >= result.numel()) {
        //   IJ -= result.numel() - 1
        //   if (IJ == W) {
        //     IJ = 0
        //   }
        // }

        i = Math.floor(IJ / W)
        j = IJ % W
    } else {
        oldi = Math.floor(IJ / D2)
        oldj = IJ % D2

        IJ = (IJ + 1) % result2.numel()
        // HEY
        // IJ += D2
        // if (IJ >= result2.numel()) {
        //   IJ -= result2.numel() - 1
        //   if (IJ == D2) {
        //     IJ = 0
        //   }
        // }

        i = Math.floor(IJ / D2)
        j = IJ % D2
    }

    if (IJ == 0) {
        if (PHASE == 0) {
            dotprod.fillData(0)
            left.bumpRowColor(oldi, false)
            right.bumpColumnColor(oldj, false)
        } else {
            result.fillData(0)
            result2.fillData(0)
            result.bumpRowColor(oldi, false)
            left2.bumpColumnColor(oldj, false)
            dotprod2.fillData(0)
        }
        PHASE = (PHASE + 1) % 2
    }

    // update result face
    if (PHASE == 0) {
        result.setData(i, j, result_val(i, j))

        // hilight operand row/cols
        if (oldj != j) {
            if (oldj >= 0) {
                right.bumpColumnColor(oldj, false)
            }
            right.bumpColumnColor(j, true)
        }

        if (oldi != i) {
            if (oldi >= 0) {
                left.bumpRowColor(oldi, false)
            }
            left.bumpRowColor(i, true)
        }

        // move and recolor dot product vector
        dotprod.points.position.x = right.points.geometry.attributes.position.array[j * 3]
        dotprod.points.position.y = -left.points.geometry.attributes.position.array[Math.floor(IJ * D / W) * 3 + 1]
        for (let z = 0; z < dotprod.numel(); z++) {
            dotprod.setData(0, z, dotprod_val(i, j, z))
        }
    } else {
        result2.setData(i, j, result2_mlp_val(i, j))

        // hilight operand row/cols
        if (oldj != j) {
            if (oldj >= 0) {
                left2.bumpColumnColor(oldj, false)
            }
            left2.bumpColumnColor(j, true)
        }

        if (oldi != i) {
            if (oldi >= 0) {
                result.bumpColumnColor(oldi, false)
            }
            result.bumpColumnColor(i, true)
        }

        // move and recolor dot product vector
        dotprod2.points.position.z = left2.points.geometry.attributes.position.array[j * 3] + D + 1
        dotprod2.points.position.x = result.points.geometry.attributes.position.array[Math.floor(IJ * W / D2) * 3 + 1]
        for (let z = 0; z < dotprod2.numel(); z++) {
            dotprod2.setData(0, z, dotprod2_mlp_val(i, j, z))
        }
    }
}

export function createMLPDoubleMatMul() {
    const mm2 = new THREE.Group()

    // HEY
    const left_init = (y, x) => squeeze(y / H)
    // const left_init = (y, x) => squeeze(x / D)
    // const left_init = (y, x) => Math.random()
    left = new Mat(H, D, left_init).faceLeft()
    left.points.position.x = -1
    mm2.add(left.points)

    // HEY
    // const right_init = (y, x) => squeeze(y / D)
    const right_init = (y, x) => squeeze(x / W)
    // const right_init = (y, x) => Math.random()
    right = new Mat(D, W, right_init).faceUp()
    right.points.position.y = 1
    mm2.add(right.points)

    result = new Mat(H, W, result_val).faceFront()
    result.points.position.z = D
    result.fillData(0)
    mm2.add(result.points)

    const dotprod_init = (y, x) => dotprod_val(0, 0, x)
    dotprod = new Mat(1, D, dotprod_init).faceLeft()
    mm2.add(dotprod.points)

    // HEY
    // const right2_init = (y, x) => squeeze(y / W)
    const left2_init = (y, x) => squeeze(x / D2)
    // const right2_init = (y, x) => Math.random()
    left2 = new Mat(H, D2, left2_init).faceLeft()
    left2.points.rotation.x = -Math.PI / 2
    left2.points.rotation.z = -Math.PI / 2
    left2.points.position.x = W
    // left2.points.position.y = -H
    left2.points.position.z = D + 1
    mm2.add(left2.points)

    result2 = new Mat(W, D2, result2_mlp_val)
    result2.points.rotation.x = -Math.PI / 2
    // result2.points.rotation.x = Math.PI
    result2.points.rotation.z = -Math.PI / 2
    // result2.points.position.x = W
    result2.points.position.y = -H
    result2.points.position.z = D + 1
    result2.fillData(0)
    mm2.add(result2.points)

    const dotprod2_init = (y, x) => dotprod2_mlp_val(0, 0, x)
    dotprod2 = new Mat(1, H, dotprod2_init)
    dotprod2.points.rotation.z = -Math.PI / 2
    mm2.add(dotprod2.points)

    // center cube on 0,0,0
    mm2.position.x = -W / 2
    mm2.position.y = H / 2
    mm2.position.z = -(D + D2) / 2

    // HEY
    IJ = result.numel() - 1

    add_bump(bump_mlp_mm2)

    return mm2
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
    // scene.add(createMatMul())
    // scene.add(createAttnDoubleMatMul())
    scene.add(createMLPDoubleMatMul())

    return scene
}


