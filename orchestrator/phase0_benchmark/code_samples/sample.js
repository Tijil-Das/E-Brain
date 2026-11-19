'use strict';
const { promises: fs } = require('fs');
const path = require('path');

class LRUCache {
  constructor(maxSize) {
    this._max  = maxSize;
    this._map  = new Map();
  }
  get(key) {
    if (!this._map.has(key)) return undefined;
    const val = this._map.get(key);
    this._map.delete(key);
    this._map.set(key, val);
    return val;
  }
  set(key, val) {
    if (this._map.has(key)) this._map.delete(key);
    if (this._map.size >= this._max)
      this._map.delete(this._map.keys().next().value);
    this._map.set(key, val);
  }
}

async function readJsonLines(filePath) {
  const text  = await fs.readFile(filePath, 'utf8');
  return text.split('\n').filter(Boolean).map(JSON.parse);
}

module.exports = { LRUCache, readJsonLines };
