"use strict"

import { GUI } from 'lil-gui'
import * as viz from './viz.js'
import * as util from './util.js'

let gui // global! we manage reinitialization, disposal etc.

export function initGui(params, callbacks, info) {
  const { initObj, getObj, saveUrl, updateTitle, animPause, animStep } = callbacks
  const { url_info, render_info } = info

  function set(k, v, f = initObj, param_path = p => p, obj_path = o => o) {
    param_path(params)[k] = v
    if (obj_path(getObj()).params[k] != v) {
      f(v)
    }
  }

  function addNumParam(gui, k, min, max, f = initObj, param_path = p => p, obj_path = o => o) {
    return gui.add(param_path(params), k, min, max).onChange(
      x => set(k, x, f, param_path, obj_path)
    )
  }

  function addIntParam(gui, k, min, max, f = initObj, param_path = p => p, obj_path = o => o) {
    return gui.add(param_path(params), k, min, max).onChange(
      x => set(k, Math.floor(x), f, param_path, obj_path)
    )
  }

  function addChoiceParam(gui, k, choices, f = initObj, param_path = p => p, obj_path = o => o) {
    return gui.add(param_path(params), k, choices).onChange(
      x => set(k, x, f, param_path, obj_path)
    )
  }

  function addParam(gui, k, f = initObj, param_path = p => p, obj_path = o => o) {
    return gui.add(param_path(params), k).onChange(
      x => set(k, x, f, param_path, obj_path)
    )
  }

  const findController = (gui, x) =>
    gui.controllers.find(c => c.property == x)

  const findFolder = (g, title) =>
    g.folders.find(f => f._title == title)

  const findFolders = (g, title, recursive = false) =>
    (recursive ? g.foldersRecursive() : g.folders).filter(f => f._title == title)

  function clearFolder(g) {
    while (g.folders.length > 0) {
      g.folders[0].destroy()
    }
    while (g.controllers.length > 0) {
      g.controllers[0].destroy()
    }
  }

  function addFolder(parent, name, p) {
    const child = parent.addFolder(name).open(p.folder == 'open')
    child.onOpenClose(g => {
      if (g === child) {
        p.folder = g._closed ? 'closed' : 'open'
        saveUrl()
      }
    })
    return child
  }

  // layout schemes

  function syncLayoutSchemeAndInit() {
    viz.setLayoutScheme(params)
    const is_custom = params.layout.scheme == 'custom'
    findFolders(findFolder(gui, 'left'), 'layout', true).map(f => f.show(is_custom))
    findFolders(findFolder(gui, 'right'), 'layout', true).map(f => f.show(is_custom))
    initObj()
  }

  // child mat/matmul state change 

  const childInit = (parent_init, left_child, left_parent) =>
    left_child == left_parent ? parent_init :
      parent_init.slice(0, 3) == 'row' ? ('col' + parent_init.slice(3)) :
        parent_init.slice(0, 3) == 'col' ? ('row' + parent_init.slice(3)) :
          parent_init

  const childMat = (p, depth, left_child, left_parent) => {
    return {
      name: p.name + (left_child ? 'L' : 'R'),
      matmul: false,
      h: left_child ? p.h : depth,
      w: left_child ? depth : p.w,
      init: childInit(p.init, left_child, left_parent), // p.init,
      url: p.url,
      expr: p.expr,
      min: p.min,
      max: p.max,
      dropout: p.dropout,
    }
  }

  function syncChildParams(g, path, ancestors) {
    const height = p => p.matmul ? height(p.left) : p.h
    const width = p => p.matmul ? width(p.right) : p.w
    const p = path(params)
    const pp = ancestors[0](params)
    clearFolder(g)
    if (p.matmul) {
      const is_left = p === pp.left
      const depth = is_left ? width(pp.right) : height(pp.left)
      const rule = viz.LAYOUT_RULES[params.layout.scheme]
      const layoutProto = left_child => rule ?
        viz.childLayout(pp.layout, rule, left_child, p === pp.left) :
        util.copyTree(pp.layout)
      util.updateProps(p, {
        epilog: pp.epilog,
        left: childMat(p, depth, true, p === pp.left),
        right: childMat(p, depth, false, p === pp.left),
        anim: viz.defaultAnim(),
        layout: layoutProto(pp, is_left),
      })
      util.deleteProps(p, ['h', 'w', 'init', 'url', 'expr', 'min', 'max', 'dropout'])
      addMatmulParams(g, path, ancestors)
    } else {
      util.updateProps(p, {
        h: height(p.left),
        w: width(p.right),
        init: p === pp.left ? viz.leftLeaf(p).init : viz.rightLeaf(p).init,
        url: p === pp.left ? viz.leftLeaf(p).url : viz.rightLeaf(p).url,
        expr: p === pp.left ? viz.leftLeaf(p).expr : viz.rightLeaf(p).expr,
        min: -1,
        max: 1,
        dropout: 0,
      })
      util.deleteProps(p, ['left', 'right', 'epilog', 'anim', 'block', 'layout'])
      addMatParams(g, path, ancestors)
    }
    params.expr = viz.genExpr(params)
  }

  // addMatParams

  function addMatParams(g, path, ancestors) {
    const p = path(params)

    addParam(g, 'name', x => {
      if (x.length > 0) {
        path(getObj()).setName(x)
        updateTitle()
        params.expr = viz.genExpr(params)
        saveUrl()
      } else {
        p.name = path(getObj()).params.name
      }
    }, path, path).listen()

    addParam(g, 'matmul', v => {
      syncChildParams(g, path, ancestors)
      syncLayoutSchemeAndInit()
    }, path, path)

    addIntParam(g, 'h', 1, 1024, x => {
      viz.fixShape(p.h, p.w, p, ancestors, params)
      initObj()
    }, path, path).listen()

    addIntParam(g, 'w', 1, 1024, x => {
      viz.fixShape(p.h, p.w, p, ancestors, params)
      initObj()
    }, path, path).listen()

    addChoiceParam(g, 'init', viz.INITS, v => {
      findController(g, 'url').show(v == 'url')
      findController(g, 'expr').show(v == 'expr')
      findController(g, 'min').show(viz.useRange(v))
      findController(g, 'max').show(viz.useRange(v))
      findController(g, 'dropout').show(viz.useDropout(v))
      initObj()
    }, path).listen()

    p.url ||= '' // temp BC
    g.add(p, 'url').onFinishChange(url => {
      p.url = url
      initObj()
    }).show(p.init == 'url')
    p.expr ||= '' // temp BC
    g.add(p, 'expr').onFinishChange(expr => {
      p.expr = expr
      initObj()
    }).show(p.init == 'expr')
    addNumParam(g, 'min', -1.0, 1.0, initObj, path, path).show(viz.useRange(p.init))
    addNumParam(g, 'max', 0.0, 1.0, initObj, path, path).show(viz.useRange(p.init))
    addNumParam(g, 'dropout', 0.0, 1.0, initObj, path, path).show(viz.useDropout(p.init))
  }

  // addMatmulParams

  function addMatmulParams(g, path = x => x, ancestors = []) {
    const p = path(params)
    const is_root = ancestors.length == 0

    if (is_root) {
      g.add(p, 'expr').onFinishChange(evalExpr).listen()
    }

    addParam(g, 'name', x => {
      if (x.length > 0) {
        path(getObj()).setName(x)
        updateTitle()
        params.expr = viz.genExpr(params)
        saveUrl()
      } else {
        p.name = path(getObj()).params.name
      }
    }, path, path).listen()

    if (!is_root) {
      addParam(g, 'matmul', v => {
        syncChildParams(g, path, ancestors)
        initObj()
      }, path, path)
    }

    addChoiceParam(g, 'epilog', viz.EPILOGS, initObj, path, path)

    // left/right
    // const title = name => (is_root ? '' : `${g._title} / `) + name
    // const gui_left = addFolder(g, title('left'), p.left)
    const gui_left = addFolder(g, 'left', p.left)
    const add_left = p.left.matmul ? addMatmulParams : addMatParams
    add_left(gui_left, x => path(x).left, [path].concat(ancestors))

    // const gui_right = addFolder(g, title('right'), p.right)
    const gui_right = addFolder(g, 'right', p.right)
    const add_right = p.right.matmul ? addMatmulParams : addMatParams
    add_right(gui_right, x => path(x).right, [path].concat(ancestors))

    // animation
    const gui_anim = addFolder(g, 'animation', p.anim)
    if (is_root) {
      addChoiceParam(gui_anim, 'alg', viz.TOP_LEVEL_ANIM_ALGS, initObj, p => p.anim)
      addIntParam(gui_anim, 'speed', 1, 100, _ => { }, p => p.anim)
      gui_anim.add({ pause: false }, 'pause').onChange(x => animPause(x))
      gui_anim.add({ step: animStep }, 'step')
      addChoiceParam(gui_anim, 'fuse', viz.FUSE_MODE, initObj, p => p.anim)
      addParam(gui_anim, 'hide inputs', x => getObj().hideInputs(x), p => p.anim)
      params.anim.spin ||= params.deco.spin || 0 // temp BC
      addNumParam(gui_anim, 'spin', -10, 10, x => { }, p => p.anim)
    } else {
      addChoiceParam(gui_anim, 'alg', viz.ANIM_ALGS, initObj, p => path(p).anim, path)
    }

    // blocking
    p.block ||= viz.defaultBlock() // temp BC
    const gui_block = addFolder(g, 'blocking', p.block)
    if (is_root) {
      addIntParam(gui_block, 'i blocks', 1, 16, initObj, p => p.block)
      addIntParam(gui_block, 'k blocks', 1, 16, initObj, p => p.block)
      addIntParam(gui_block, 'j blocks', 1, 16, initObj, p => p.block)
    } else {
      if (p.block['j blocks']) { // temp BC
        p.block['k blocks'] = p.block['j blocks']
        delete p.block['j blocks']
      }
      addIntParam(gui_block, 'k blocks', 1, 32, initObj, p => path(p).block, path)
    }

    // layout
    const gui_layout = addFolder(g, 'layout', p.layout)
    if (is_root) {
      addNumParam(gui_layout, 'gap', 1, 64, initObj, p => p.layout)
      addNumParam(gui_layout, 'scatter', 0, 128, initObj, p => p.layout)
      addIntParam(gui_layout, 'molecule', 1, 8, initObj, p => p.layout)
      addNumParam(gui_layout, 'blast', -2.0, 2.0, initObj, p => p.layout)

      addChoiceParam(gui_layout, 'scheme', viz.SCHEMES, syncLayoutSchemeAndInit, p => p.layout).listen()
      addChoiceParam(gui_layout, 'polarity', viz.POLARITIES, syncLayoutSchemeAndInit, p => path(p).layout, path)
      addChoiceParam(gui_layout, 'left placement', viz.LEFT_PLACEMENTS, syncLayoutSchemeAndInit, p => path(p).layout, path)
      addChoiceParam(gui_layout, 'right placement', viz.RIGHT_PLACEMENTS, syncLayoutSchemeAndInit, p => path(p).layout, path)
      addChoiceParam(gui_layout, 'result placement', viz.RESULT_PLACEMENTS, syncLayoutSchemeAndInit, p => path(p).layout, path)
    } else {
      gui_layout.show(params.layout.scheme == 'custom')
      addChoiceParam(gui_layout, 'polarity', viz.POLARITIES, initObj, p => path(p).layout, path)
      addChoiceParam(gui_layout, 'left placement', viz.LEFT_PLACEMENTS, initObj, p => path(p).layout, path)
      addChoiceParam(gui_layout, 'right placement', viz.RIGHT_PLACEMENTS, initObj, p => path(p).layout, path)
      addChoiceParam(gui_layout, 'result placement', viz.RESULT_PLACEMENTS, initObj, p => path(p).layout, path)
    }
  }

  // expr eval

  let prev_expr = params.expr

  function evalExpr(e) {
    params.expr = e
    if (!viz.syncExpr(params)) {
      params.expr = prev_expr
    } else {
      prev_expr = e

      const gui_left = findFolder(gui, 'left')
      clearFolder(gui_left)
      const add_left = params.left.matmul ? addMatmulParams : addMatParams
      add_left(gui_left, x => x.left, [x => x])

      const gui_right = findFolder(gui, 'right')
      clearFolder(gui_right)
      const add_right = params.right.matmul ? addMatmulParams : addMatParams
      add_right(gui_right, x => x.right, [x => x])

      syncLayoutSchemeAndInit()
    }
    saveUrl()  // global onFinishChange front-runs
  }

  params.deco.shape ||= false // temp BC
  params.deco['lens size'] ||= 0.25 // temp BC
  params.deco.magnification ||= 5 // temp BC
  function addDecoParams(g) {
    const gui_deco = addFolder(g, 'deco', params.deco)
    addNumParam(gui_deco, 'legends', 0, 10, x => getObj().setLegends(x), p => p.deco)
    addParam(gui_deco, 'shape', x => getObj().setLegends(undefined, x), p => p.deco)
    addNumParam(gui_deco, 'row guides', 0.0, 1.0, x => getObj().setRowGuides(x), p => p.deco)
    addNumParam(gui_deco, 'flow guides', 0.0, 1.0, x => getObj().setFlowGuide(x), p => p.deco)
    addIntParam(gui_deco, 'spotlight', 0, 10, x => getObj().updateLabels(params), p => p.deco)
    addNumParam(gui_deco, 'lens size', 0.0, 1.0, x => { }, p => p.deco)
    addNumParam(gui_deco, 'magnification', 1, 25, x => { }, p => p.deco)
    addParam(gui_deco, 'interior spotlight', x => getObj().updateLabels(params), p => p.deco)
    // addParam(gui_deco, 'axes', x => initAxes(getObj().params.axes = x), p => p.deco)
  }

  params.viz['elem scale'] ||= 1.25 // temp BC
  function addVizParams(g) {
    const gui_viz = addFolder(g, 'colors and sizes', params.viz)
    addChoiceParam(gui_viz, 'sensitivity', viz.SENSITIVITIES, initObj, p => p.viz)
    addNumParam(gui_viz, 'min size', 0.0, 1.0, initObj, p => p.viz)
    addNumParam(gui_viz, 'min light', 0.0, 1.0, initObj, p => p.viz)
    addNumParam(gui_viz, 'max light', 0.0, 1.0, initObj, p => p.viz)
    addNumParam(gui_viz, 'elem scale', 0.1, 2.0, initObj, p => p.viz)
    addNumParam(gui_viz, 'zero hue', 0.0, 1.0, initObj, p => p.viz)
    addNumParam(gui_viz, 'hue gap', 0.0, 1.0, initObj, p => p.viz)
    addNumParam(gui_viz, 'hue spread', 0.0, 1.0, initObj, p => p.viz)
  }

  function addDiagParams(g) {
    const gui_diag = addFolder(g, 'diag', params.diag)
    gui_diag.add(url_info, 'json').listen()
    gui_diag.add(url_info, 'url').listen()
    gui_diag.add(url_info, 'compressed').listen()
    gui_diag.add(render_info, 'geometries').listen()
    // gui_diag.add(render_info, 'textures').listen()
    // gui_diag.add(display_info, 'x').listen()
    // gui_diag.add(display_info, 'y').listen()
  }

  // --- 

  gui && gui.destroy()
  gui = new GUI({ title: 'mm' }).open((params.folder || 'closed') == 'open')

  gui.onOpenClose(g => {
    if (g === gui) {
      params.folder = g._closed ? 'closed' : 'open'
      saveUrl()
    }
  })

  addMatmulParams(gui)
  addDecoParams(gui)
  addVizParams(gui)
  addDiagParams(gui)

  gui.onFinishChange(saveUrl)

  return gui
}