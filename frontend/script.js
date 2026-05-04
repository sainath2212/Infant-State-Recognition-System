// ===== Navbar scroll effect =====
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 50);
});

// ===== Mobile menu =====
const mobileBtn = document.getElementById('mobile-menu-btn');
const mobileMenu = document.getElementById('mobile-menu');
mobileBtn.addEventListener('click', () => {
  mobileMenu.classList.toggle('open');
  mobileBtn.classList.toggle('active');
});
mobileMenu.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', () => mobileMenu.classList.remove('open'));
});

// ===== Active nav link on scroll =====
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-link');
function updateActiveLink() {
  const scrollY = window.scrollY + 120;
  sections.forEach(section => {
    const top = section.offsetTop;
    const height = section.offsetHeight;
    const id = section.getAttribute('id');
    if (scrollY >= top && scrollY < top + height) {
      navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === '#' + id) link.classList.add('active');
      });
    }
  });
}
window.addEventListener('scroll', updateActiveLink);

// ===== Intersection Observer for reveal animations =====
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

// ===== Phase tabs =====
const phaseTabs = document.querySelectorAll('.phase-tab');
const phasePanels = document.querySelectorAll('.phase-panel');
phaseTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const phase = tab.dataset.phase;
    phaseTabs.forEach(t => t.classList.remove('active'));
    phasePanels.forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('phase-' + phase).classList.add('active');
  });
});

// ===== Image modal =====
const modal = document.getElementById('image-modal');
const modalImg = document.getElementById('modal-img');
const modalCaption = document.getElementById('modal-caption');
const modalClose = document.querySelector('.modal-close');

document.querySelectorAll('.gallery-card').forEach(card => {
  card.addEventListener('click', () => {
    const img = card.querySelector('img');
    const caption = card.querySelector('.gallery-overlay span');
    modalImg.src = img.src;
    modalImg.alt = img.alt;
    modalCaption.textContent = caption ? caption.textContent : '';
    modal.classList.add('open');
    document.body.style.overflow = 'hidden';
  });
});

modalClose.addEventListener('click', closeModal);
modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

function closeModal() {
  modal.classList.remove('open');
  document.body.style.overflow = '';
}
