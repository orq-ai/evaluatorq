'use strict';
/* A small DOM for the dashboard form tests: real markup in, the selector subset the controllers use, bubbling events. */

const VOID = new Set(['input', 'br', 'img', 'meta', 'link', 'hr']);
const ENTITIES = {amp: '&', lt: '<', gt: '>', quot: '"', '#39': "'", times: '\u00d7', ndash: '\u2013', rarr: '\u2192', hellip: '\u2026'};

function decode(text) {
  return text.replace(/&(#?\w+);/g, (whole, name) => (name in ENTITIES ? ENTITIES[name] : whole));
}

class Classes {
  constructor(element) { this.element = element; }
  list() { return (this.element.getAttribute('class') || '').split(/\s+/).filter(Boolean); }
  contains(name) { return this.list().includes(name); }
  add(name) { if (!this.contains(name)) this.element.setAttribute('class', [...this.list(), name].join(' ')); }
  remove(name) { this.element.setAttribute('class', this.list().filter(item => item !== name).join(' ')); }
  toggle(name, force) {
    const on = force === undefined ? !this.contains(name) : Boolean(force);
    if (on) this.add(name); else this.remove(name);
    return on;
  }
}

function parseCompound(text) {
  const compound = {tag: null, id: null, classes: [], attrs: [], checked: false};
  const pattern = /^([a-zA-Z][\w-]*)|#([\w-]+)|\.([\w-]+)|\[([\w-]+)(?:([\^]?=)"([^"]*)")?\]|:(checked)/y;
  let rest = text;
  let index = 0;
  while (index < rest.length) {
    pattern.lastIndex = index;
    const m = pattern.exec(rest);
    if (!m) throw new Error('unsupported selector: ' + text);
    if (m[1]) compound.tag = m[1];
    else if (m[2]) compound.id = m[2];
    else if (m[3]) compound.classes.push(m[3]);
    else if (m[4]) compound.attrs.push({name: m[4], op: m[5] || null, value: m[6]});
    else if (m[7]) compound.checked = true;
    index = pattern.lastIndex;
  }
  return compound;
}

function parseSelector(selector) {
  return selector.split(',').map(part => {
    const tokens = part.trim().match(/(?:[^\s\[]|\[[^\]]*\])+/g) || [];
    return tokens.map(parseCompound);
  });
}

class Node {
  constructor() { this.parent = null; this.children = []; }
}

class Text extends Node {
  constructor(text) { super(); this.text = text; }
  get textContent() { return this.text; }
}

class Element extends Node {
  constructor(tag, attrs, doc) {
    super();
    this.tagName = tag;
    this.attrs = new Map(attrs);
    this.ownerDocument = doc;
    this.listeners = {};
    this._value = undefined;
    this._checked = undefined;
    this.files = [];
    if (tag === 'textarea') this._value = undefined;
    const self = this;
    this.dataset = new Proxy({}, {
      get(_, key) { return self.getAttribute('data-' + String(key).replace(/[A-Z]/g, c => '-' + c.toLowerCase())); },
      set(_, key, value) { self.setAttribute('data-' + String(key).replace(/[A-Z]/g, c => '-' + c.toLowerCase()), value); return true; },
    });
    this.classList = new Classes(this);
  }

  getAttribute(name) { return this.attrs.has(name) ? this.attrs.get(name) : null; }
  setAttribute(name, value) { this.attrs.set(name, String(value)); }
  removeAttribute(name) { this.attrs.delete(name); }
  hasAttribute(name) { return this.attrs.has(name); }

  get id() { return this.getAttribute('id') || ''; }
  get name() { return this.getAttribute('name') || ''; }
  get hidden() { return this.hasAttribute('hidden'); }
  set hidden(on) { if (on) this.setAttribute('hidden', ''); else this.removeAttribute('hidden'); }
  get disabled() { return this.hasAttribute('disabled'); }
  set disabled(on) { if (on) this.setAttribute('disabled', ''); else this.removeAttribute('disabled'); }
  get type() {
    if (this.tagName === 'textarea') return 'textarea';
    if (this.tagName === 'select') return 'select-one';
    return this.getAttribute('type') || (this.tagName === 'button' ? 'submit' : 'text');
  }
  get checked() { return this._checked === undefined ? this.hasAttribute('checked') : this._checked; }
  set checked(on) { this._checked = Boolean(on); }
  get value() {
    if (this._value !== undefined) return this._value;
    if (this.tagName === 'textarea') return this.textContent;
    if (this.tagName === 'select') {
      const option = this.querySelector('option');
      return option ? option.getAttribute('value') : '';
    }
    const attr = this.getAttribute('value');
    if (attr !== null) return attr;
    return this.type === 'checkbox' || this.type === 'radio' ? 'on' : '';
  }
  set value(next) { this._value = String(next); }

  get textContent() { return this.children.map(child => child.textContent).join(''); }
  set textContent(text) { this.children = [new Text(String(text))]; this.children[0].parent = this; }

  set innerHTML(html) {
    this.children = [];
    for (const child of parseFragment(html, this.ownerDocument)) this._append(child);
  }
  get innerHTML() { return this.children.map(child => (child instanceof Text ? child.text : child.outerHTML)).join(''); }
  get outerHTML() {
    const attrs = [...this.attrs].map(([k, v]) => ' ' + k + '="' + v + '"').join('');
    return '<' + this.tagName + attrs + '>' + (VOID.has(this.tagName) ? '' : this.innerHTML + '</' + this.tagName + '>');
  }
  insertAdjacentHTML(where, html) {
    if (where !== 'afterbegin') throw new Error('unsupported position');
    const nodes = parseFragment(html, this.ownerDocument);
    nodes.forEach(node => { node.parent = this; });
    this.children = [...nodes, ...this.children];
  }
  _append(child) { child.parent = this; this.children.push(child); }

  descendants() {
    const out = [];
    const walk = node => node.children.forEach(child => { if (child instanceof Element) { out.push(child); walk(child); } });
    walk(this);
    return out;
  }
  matchesCompound(compound) {
    if (compound.tag && this.tagName !== compound.tag) return false;
    if (compound.id && this.id !== compound.id) return false;
    if (!compound.classes.every(name => this.classList.contains(name))) return false;
    if (compound.checked && !this.checked) return false;
    return compound.attrs.every(({name, op, value}) => {
      const actual = this.getAttribute(name);
      if (actual === null) return false;
      if (op === '=') return actual === value;
      if (op === '^=') return actual.startsWith(value);
      return true;
    });
  }
  matchesChain(chain) {
    if (!this.matchesCompound(chain[chain.length - 1])) return false;
    let node = this.parent;
    for (let i = chain.length - 2; i >= 0; i -= 1) {
      while (node && !(node instanceof Element && node.matchesCompound(chain[i]))) node = node.parent;
      if (!node) return false;
      node = node.parent;
    }
    return true;
  }
  matches(selector) { return parseSelector(selector).some(chain => this.matchesChain(chain)); }
  closest(selector) {
    let node = this;
    while (node instanceof Element) {
      if (node.matches(selector)) return node;
      node = node.parent;
    }
    return null;
  }
  querySelectorAll(selector) {
    const chains = parseSelector(selector);
    return this.descendants().filter(element => chains.some(chain => element.matchesChain(chain)));
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }

  addEventListener(type, callback) { (this.listeners[type] = this.listeners[type] || []).push(callback); }
  dispatchEvent(event) {
    if (!event.target) event.target = this;
    event.preventDefault = event.preventDefault || (() => { event.defaultPrevented = true; });
    let node = this;
    while (node instanceof Element) {
      for (const callback of node.listeners[event.type] || []) callback(event);
      if (!event.bubbles) break;
      node = node.parent;
    }
    return !event.defaultPrevented;
  }
  focus() { this.ownerDocument.activeElement = this; }
  click() {
    if (this.type === 'radio') {
      const root = this.closest('form') || this.ownerDocument.root;
      root.querySelectorAll('input[name="' + this.name + '"]').forEach(other => { other._checked = false; });
      this._checked = true;
    } else if (this.type === 'checkbox') {
      this._checked = !this.checked;
    }
    this.dispatchEvent({type: 'click', bubbles: true, target: this});
    if (this.type === 'radio' || this.type === 'checkbox') this.dispatchEvent({type: 'change', bubbles: true, target: this});
    if (this.tagName === 'button' && this.type === 'submit') {
      const form = this.closest('form');
      if (form) form.dispatchEvent({type: 'submit', bubbles: true, target: form});
    }
  }
  replaceWith(next) {
    const siblings = this.parent.children;
    siblings[siblings.indexOf(this)] = next;
    next.parent = this.parent;
    this.parent = null;
    const doc = this.ownerDocument;
    [next, ...next.descendants()].forEach(element => { element.ownerDocument = doc; });
  }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this);
    this.parent = null;
  }
}

function parseFragment(html, doc) {
  const root = new Element('#root', [], doc);
  const stack = [root];
  const tokens = /<!--[\s\S]*?-->|<\/([a-zA-Z][\w-]*)\s*>|<([a-zA-Z][\w-]*)((?:\s+[^\s=\/>]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))?)*)\s*\/?>|([^<]+)/g;
  const attrPattern = /([^\s=\/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g;
  let m;
  while ((m = tokens.exec(html))) {
    if (m[1]) {
      for (let i = stack.length - 1; i > 0; i -= 1) {
        if (stack[i].tagName === m[1].toLowerCase()) { stack.length = i; break; }
      }
    } else if (m[2]) {
      const tag = m[2].toLowerCase();
      const attrs = [];
      let a;
      attrPattern.lastIndex = 0;
      while ((a = attrPattern.exec(m[3] || ''))) attrs.push([a[1], decode(a[2] ?? a[3] ?? a[4] ?? '')]);
      const element = new Element(tag, attrs, doc);
      stack[stack.length - 1]._append(element);
      if (tag === 'textarea') {
        const end = html.indexOf('</textarea>', tokens.lastIndex);
        element._append(new Text(decode(html.slice(tokens.lastIndex, end))));
        tokens.lastIndex = end + '</textarea>'.length;
      } else if (!VOID.has(tag) && !m[0].endsWith('/>')) {
        stack.push(element);
      }
    } else if (m[4] !== undefined) {
      stack[stack.length - 1]._append(new Text(decode(m[4])));
    }
  }
  const out = root.children;
  out.forEach(node => { node.parent = null; });
  return out;
}

class Document {
  constructor(html) {
    this.root = new Element('#root', [], this);
    this.activeElement = null;
    this.readyState = 'complete';
    this.listeners = {};
    if (html !== undefined) {
      for (const child of parseFragment(html, this)) this.root._append(child);
    }
  }
  getElementById(id) { return this.root.descendants().find(element => element.id === id) || null; }
  querySelector(selector) { return this.root.querySelector(selector); }
  querySelectorAll(selector) { return this.root.querySelectorAll(selector); }
  addEventListener(type, callback) { (this.listeners[type] = this.listeners[type] || []).push(callback); }
}

class DOMParser {
  parseFromString(html) { return new Document(html); }
}

module.exports = {Document, DOMParser};
