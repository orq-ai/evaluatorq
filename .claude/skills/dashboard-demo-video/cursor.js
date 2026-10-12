// Demo-only: draw a small white fake cursor and a thin fading ring on every click, so headless recordings show where the click went.
(() => {
  const add = () => {
    if (document.getElementById('__demo_cursor')) return;
    const st = document.createElement('style');
    st.textContent = `#__demo_cursor{position:fixed;z-index:2147483647;width:14px;height:14px;margin:-7px 0 0 -7px;box-sizing:border-box;border-radius:50%;background:#fff;border:1.5px solid #111;box-shadow:0 1px 4px rgba(0,0,0,.35);pointer-events:none;transition:left .5s ease,top .5s ease;left:-50px;top:-50px}
#__demo_cursor.__demo_press{animation:__demo_p .25s ease-out}
@keyframes __demo_p{40%{transform:scale(.65)}}
.__demo_ripple{position:fixed;z-index:2147483646;width:14px;height:14px;margin:-7px 0 0 -7px;box-sizing:border-box;border-radius:50%;border:1.5px solid rgba(255,255,255,.95);box-shadow:0 0 0 1px rgba(0,0,0,.5),inset 0 0 0 1px rgba(0,0,0,.5);pointer-events:none;animation:__demo_r .4s ease-out forwards}
@keyframes __demo_r{to{transform:scale(3.2);opacity:0}}`;
    document.head.appendChild(st);
    const c = document.createElement('div'); c.id = '__demo_cursor'; document.body.appendChild(c);
    const move = e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; };
    addEventListener('mousemove', move, true);
    addEventListener('mousedown', e => {
      move(e);
      c.classList.remove('__demo_press'); void c.offsetWidth; c.classList.add('__demo_press');
      const r = document.createElement('div'); r.className = '__demo_ripple'; r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px'; document.body.appendChild(r); setTimeout(() => r.remove(), 450);
    }, true);
  };
  if (document.body) add(); else addEventListener('DOMContentLoaded', add);
})();
