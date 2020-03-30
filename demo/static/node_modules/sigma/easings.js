"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.cubicInOut = exports.cubicOut = exports.cubicIn = exports.quadraticInOut = exports.quadraticOut = exports.quadraticIn = exports.linear = void 0;

/**
 * Sigma.js Easings
 * =================
 *
 * Handy collection of easing functions.
 */
var linear = function linear(k) {
  return k;
};

exports.linear = linear;

var quadraticIn = function quadraticIn(k) {
  return k * k;
};

exports.quadraticIn = quadraticIn;

var quadraticOut = function quadraticOut(k) {
  return k * (2 - k);
};

exports.quadraticOut = quadraticOut;

var quadraticInOut = function quadraticInOut(k) {
  if ((k *= 2) < 1) return 0.5 * k * k;
  return -0.5 * (--k * (k - 2) - 1);
};

exports.quadraticInOut = quadraticInOut;

var cubicIn = function cubicIn(k) {
  return k * k * k;
};

exports.cubicIn = cubicIn;

var cubicOut = function cubicOut(k) {
  return --k * k * k + 1;
};

exports.cubicOut = cubicOut;

var cubicInOut = function cubicInOut(k) {
  if ((k *= 2) < 1) return 0.5 * k * k * k;
  return 0.5 * ((k -= 2) * k * k + 2);
};

exports.cubicInOut = cubicInOut;