(() => {
  const groups = [...document.querySelectorAll('.lang-tabs')];
  if (!groups.length) return;
  let language = 'python';
  try { language = localStorage.getItem('kap-language') || language; } catch (_) {}
  if (!['python', 'rust'].includes(language)) language = 'python';
  const controls = [];
  function select(value) {
    language = value;
    try { localStorage.setItem('kap-language', value); } catch (_) {}
    controls.forEach(({ blocks, buttons }) => {
      for (const name of ['python', 'rust']) {
        blocks[name].hidden = name !== value;
        buttons[name].setAttribute('aria-pressed', String(name === value));
      }
    });
  }
  for (const group of groups) {
    const blocks = {};
    for (const name of ['python', 'rust']) {
      const code = group.querySelector(`code.language-${name}`);
      if (code) blocks[name] = code.closest('pre');
    }
    if (!blocks.python || !blocks.rust) continue;
    const bar = document.createElement('div');
    bar.className = 'language-buttons';
    bar.setAttribute('role', 'group');
    bar.setAttribute('aria-label', 'Example language');
    const buttons = {};
    for (const name of ['python', 'rust']) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = name === 'python' ? 'Python' : 'Rust';
      button.addEventListener('click', () => select(name));
      bar.append(button);
      buttons[name] = button;
    }
    group.prepend(bar);
    controls.push({ blocks, buttons });
  }
  select(language);
})();
