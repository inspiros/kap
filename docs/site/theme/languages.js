(() => {
  const menu = document.querySelector('#mdbook-menu-bar');
  if (!menu) return;
  const groups = [...document.querySelectorAll('.lang-tabs')].map(group => {
    const blocks = {};
    for (const name of ['python', 'rust']) {
      const code = group.querySelector(`code.language-${name}`);
      if (code) blocks[name] = code.closest('pre');
    }
    return blocks;
  }).filter(blocks => blocks.python && blocks.rust);
  let language = 'python';
  try { language = localStorage.getItem('kap-language') || language; } catch (_) {}
  if (!['python', 'rust'].includes(language)) language = 'python';
  const bar = document.createElement('div');
  bar.className = 'language-buttons';
  bar.setAttribute('role', 'group');
  bar.setAttribute('aria-label', 'Example language');
  const buttons = {};
  function select(value) {
    language = value;
    try { localStorage.setItem('kap-language', value); } catch (_) {}
    for (const name of ['python', 'rust']) {
      buttons[name].setAttribute('aria-pressed', String(name === value));
      groups.forEach(blocks => { blocks[name].hidden = name !== value; });
    }
  }
  for (const name of ['python', 'rust']) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = name === 'python' ? 'Python' : 'Rust';
    button.title = `Show ${name === 'python' ? 'Python' : 'Rust'} examples across the documentation`;
    button.addEventListener('click', () => select(name));
    bar.append(button);
    buttons[name] = button;
  }
  menu.insertBefore(bar, menu.querySelector('.right-buttons'));
  select(language);
})();
