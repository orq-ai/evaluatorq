// ⌘K / Ctrl+K focuses the Material search box (Material binds /, s, f natively).
document.addEventListener("keydown", function (e) {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    var input = document.querySelector(".md-search__input");
    if (input) {
      e.preventDefault();
      var toggle = document.querySelector("[data-md-toggle=search]");
      if (toggle && !toggle.checked) toggle.checked = true; // open drawer on mobile
      input.focus();
    }
  }
});
