"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.floatColor = floatColor;
exports.matrixFromCamera = matrixFromCamera;
exports.extractPixel = extractPixel;
exports.canUse32BitsIndices = canUse32BitsIndices;

var _matrices = require("./matrices");

/**
 * Sigma.js WebGL Renderer Utils
 * ==============================
 *
 * Miscelleanous helper functions used by sigma's WebGL renderer.
 */

/**
 * Memoized function returning a float-encoded color from various string
 * formats describing colors.
 */
var FLOAT_COLOR_CACHE = {};
var INT8 = new Int8Array(4);
var INT32 = new Int32Array(INT8.buffer, 0, 1);
var FLOAT32 = new Float32Array(INT8.buffer, 0, 1);
var RGBA_TEST_REGEX = /^\s*rgba?\s*\(/;
var RGBA_EXTRACT_REGEX = /^\s*rgba?\s*\(\s*([0-9]*)\s*,\s*([0-9]*)\s*,\s*([0-9]*)(?:\s*,\s*(.*)?)?\)\s*$/;

function floatColor(val) {
  // If the color is already computed, we yield it
  if (typeof FLOAT_COLOR_CACHE[val] !== 'undefined') return FLOAT_COLOR_CACHE[val];
  var r = 0,
      g = 0,
      b = 0,
      a = 1; // Handling hexadecimal notation

  if (val[0] === '#') {
    if (val.length === 4) {
      r = parseInt(val.charAt(1) + val.charAt(1), 16);
      g = parseInt(val.charAt(2) + val.charAt(2), 16);
      b = parseInt(val.charAt(3) + val.charAt(3), 16);
    } else {
      r = parseInt(val.charAt(1) + val.charAt(2), 16);
      g = parseInt(val.charAt(3) + val.charAt(4), 16);
      b = parseInt(val.charAt(5) + val.charAt(6), 16);
    }
  } // Handling rgb notation
  else if (RGBA_TEST_REGEX.test(val)) {
      var match = val.match(RGBA_EXTRACT_REGEX);
      r = +match[1];
      g = +match[2];
      b = +match[3];
      if (match[4]) a = +match[4];
    }

  a = a * 255 | 0;
  var bits = (a << 24 | b << 16 | g << 8 | r) & 0xfeffffff;
  INT32[0] = bits;
  var color = FLOAT32[0];
  FLOAT_COLOR_CACHE[val] = color;
  return color;
}
/**
 * Function returning a matrix from the current state of the camera.
 */
// TODO: it's possible to optimize this drastically!


function matrixFromCamera(state, dimensions) {
  var angle = state.angle,
      ratio = state.ratio,
      x = state.x,
      y = state.y;
  var width = dimensions.width,
      height = dimensions.height;
  var matrix = (0, _matrices.identity)();
  var smallestDimension = Math.min(width, height);
  var cameraCentering = (0, _matrices.translate)((0, _matrices.identity)(), -x, -y),
      cameraScaling = (0, _matrices.scale)((0, _matrices.identity)(), 1 / ratio),
      cameraRotation = (0, _matrices.rotate)((0, _matrices.identity)(), -angle),
      viewportScaling = (0, _matrices.scale)((0, _matrices.identity)(), 2 * (smallestDimension / width), 2 * (smallestDimension / height)); // Logical order is reversed

  (0, _matrices.multiply)(matrix, viewportScaling);
  (0, _matrices.multiply)(matrix, cameraRotation);
  (0, _matrices.multiply)(matrix, cameraScaling);
  (0, _matrices.multiply)(matrix, cameraCentering);
  return matrix;
}
/**
 * Function extracting the color at the given pixel.
 */


function extractPixel(gl, x, y, array) {
  var data = array || new Uint8Array(4);
  gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, data);
  return data;
}
/**
 * Function used to know whether given webgl context can use 32 bits indices.
 */


function canUse32BitsIndices(gl) {
  var webgl2 = typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext;
  return webgl2 || !!gl.getExtension('OES_element_index_uint');
}