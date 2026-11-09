'use strict';
const assert = require('node:assert/strict');
const vm = require('node:vm');

/* Evaluate the slice of the production source between two markers in a vm context. */
module.exports = (source, context) => (startMarker, endMarker) => {
  const start = source.indexOf(startMarker), end = source.indexOf(endMarker, start);
  assert.ok(start >= 0 && end > start, `${startMarker} is present`);
  vm.runInContext(source.slice(start, end), context);
};
