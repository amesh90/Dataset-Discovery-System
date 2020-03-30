"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.loadProgram = loadProgram;
exports.loadFragmentShader = exports.loadVertexShader = void 0;

/**
 * Sigma.js Shader Utils
 * ======================
 *
 * Code used to load sigma's shaders.
 */

/**
 * Function used to load a shader.
 */
function loadShader(type, gl, source) {
  var glType = type === 'VERTEX' ? gl.VERTEX_SHADER : gl.FRAGMENT_SHADER; // Creating the shader

  var shader = gl.createShader(glType); // Loading source

  gl.shaderSource(shader, source); // Compiling the shader

  gl.compileShader(shader); // Retrieving compilation status

  var successfullyCompiled = gl.getShaderParameter(shader, gl.COMPILE_STATUS); // Throwing if something went awry

  if (!successfullyCompiled) {
    var infoLog = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error("sigma/renderers/webgl/shaders/utils.loadShader: error while compiling the shader:\n".concat(infoLog, "\n").concat(source));
  }

  return shader;
}

var loadVertexShader = loadShader.bind(null, 'VERTEX'),
    loadFragmentShader = loadShader.bind(null, 'FRAGMENT');
exports.loadFragmentShader = loadFragmentShader;
exports.loadVertexShader = loadVertexShader;

/**
 * Function used to load a program.
 */
function loadProgram(gl, shaders) {
  var program = gl.createProgram();
  var i, l; // Attaching the shaders

  for (i = 0, l = shaders.length; i < l; i++) {
    gl.attachShader(program, shaders[i]);
  }

  gl.linkProgram(program); // Checking status

  var successfullyLinked = gl.getProgramParameter(program, gl.LINK_STATUS);

  if (!successfullyLinked) {
    gl.deleteProgram(program);
    throw new Error('sigma/renderers/webgl/shaders/utils.loadProgram: error while linking the program.');
  }

  return program;
}