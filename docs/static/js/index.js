// ScreenParse Website JavaScript

// Copy BibTeX citation to clipboard
function copyBibtex() {
  const bibtexText = document.querySelector('.citation-box code').innerText;
  navigator.clipboard.writeText(bibtexText).then(() => {
    const button = document.querySelector('.copy-button');
    const originalText = button.innerHTML;
    button.innerHTML = '<span class="icon"><i class="fas fa-check"></i></span><span>Copied!</span>';
    button.classList.add('is-success');

    setTimeout(() => {
      button.innerHTML = originalText;
      button.classList.remove('is-success');
    }, 2000);
  }).catch(err => {
    console.error('Failed to copy: ', err);
  });
}

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  });
});

// Image modal/lightbox functionality
document.addEventListener('DOMContentLoaded', function () {
  // Create modal element (class names avoid Bulma .modal-content collision)
  const modal = document.createElement('div');
  modal.className = 'image-modal';
  modal.innerHTML = `
    <div class="img-modal-backdrop"></div>
    <div class="img-modal-body">
      <button class="img-modal-close">&times;</button>
      <img src="" alt="Enlarged image">
    </div>
  `;
  document.body.appendChild(modal);

  // Add modal styles
  const style = document.createElement('style');
  style.textContent = `
    .image-modal {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: 1000;
    }
    .image-modal.active {
      display: flex;
      align-items: flex-start;
      justify-content: center;
      overflow-y: auto;
    }
    .img-modal-backdrop {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.9);
    }
    .img-modal-body {
      position: relative;
      width: 95vw;
      max-width: 1400px;
      margin: 2vh auto;
      padding-top: 40px;
      z-index: 1;
    }
    .img-modal-body img {
      display: block;
      width: 100%;
      height: auto;
      border-radius: 4px;
    }
    .img-modal-close {
      position: fixed;
      top: 12px;
      right: 20px;
      background: rgba(0, 0, 0, 0.5);
      border: none;
      color: white;
      font-size: 32px;
      cursor: pointer;
      padding: 2px 12px;
      border-radius: 4px;
      z-index: 1001;
      line-height: 1;
    }
    .img-modal-close:hover {
      background: rgba(255, 255, 255, 0.2);
      color: #fff;
    }
    .example-image img {
      cursor: pointer;
      transition: transform 0.2s ease;
    }
    .example-image img:hover {
      transform: scale(1.02);
    }
  `;
  document.head.appendChild(style);

  // Click handler for example images
  document.querySelectorAll('.example-image img').forEach(img => {
    img.addEventListener('click', function () {
      const modalImg = modal.querySelector('.img-modal-body img');
      modalImg.src = this.src;
      modalImg.alt = this.alt;
      modal.classList.add('active');
      document.body.style.overflow = 'hidden';
    });
  });

  // Close modal handlers
  modal.querySelector('.img-modal-backdrop').addEventListener('click', closeModal);
  modal.querySelector('.img-modal-close').addEventListener('click', closeModal);

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
      closeModal();
    }
  });

  function closeModal() {
    modal.classList.remove('active');
    document.body.style.overflow = '';
  }
});

// Make annotation comparison images zoomable
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.annotation-cmp-img img').forEach(function (img) {
    img.addEventListener('click', function () {
      var modal = document.querySelector('.image-modal');
      if (modal) {
        var modalImg = modal.querySelector('.img-modal-body img');
        modalImg.src = this.src;
        modalImg.alt = this.alt;
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
      }
    });
  });
});

// Dataset Ground Truth Gallery
document.addEventListener('DOMContentLoaded', function () {
  const gtImages = [
    'gt/teaser.jpg',
    'gt/www-13thda-com-603e255bbea7887e.viz.jpg',
    'gt/www-1centre-com-d57a88dd5889ae97.viz.jpg',
    'gt/www-1stsos-com-abaae049e2fb9a6b.viz.jpg',
    'gt/www-4x4offroad-in-430ecb783a8917c1.viz.jpg',
    'gt/www-accelity-com-ef78113a31113881.viz.jpg',
    'gt/www-ace-equestrian-com-9dfec5d0a22ee7e3.viz.jpg',
    'gt/www-aeroatl-org-9a16e3214ad31d3f.viz.jpg',
    'gt/www-aeva-in-1389e951c0588ad6.viz.jpg',
    'gt/www-apsi-sas-com-9b535d16b53692ca.viz.jpg',
    'gt/www-aq1systems-com-709fb2b821dc8e53.viz.jpg',
    'gt/www-astrochemical-com-236a72730160793d.viz.jpg',
    'gt/www-aveplast-eu-588b4e3a4dbb5313.viz.jpg',
    'gt/www-bbyworld-com-0b722bac4fd0aa09.viz.jpg',
    'gt/www-capriceyachtcharter-co-uk-0a25f2189cc593d0.viz.jpg',
    'gt/www-carlsbad-optometry-net-a9b457eabaa73290.viz.jpg',
    'gt/www-casadomo-com-f36948a0f00b0915.viz.jpg',
    'gt/www-chalklanehotel-com-5dd2e8c1d9205c26.viz.jpg',
    'gt/www-clearstreammedicine-com-89fd1c51a0d05895.viz.jpg',
    'gt/www-collegeworks-com-f96362b640f58d19.viz.jpg',
    'gt/www-corveleno-com-e3b21c595cd672be.viz.jpg',
    'gt/www-cpwr-com-fa74724b769cdd33.viz.jpg',
    'gt/www-crushingtigers-com-6a2770e9d88b10fb.viz.jpg',
    'gt/www-csadesign-com-883ebbb2f6aa5b5f.viz.jpg',
    'gt/www-discountif-com-11a411bb5feba554.viz.jpg',
    'gt/www-rob-swart-nl-203f449bc816332b.viz.jpg'
  ];

  const gallery = document.getElementById('gt-gallery');
  if (!gallery || gtImages.length === 0) return;

  const basePath = 'static/images/';
  const getImageSrc = (imgPath) => basePath + imgPath;
  let currentIndex = 0;

  // Build gallery DOM
  gallery.innerHTML = `
    <div class="gt-gallery-viewer">
      <button class="gt-nav gt-nav-prev" aria-label="Previous image">
        <i class="fas fa-chevron-left"></i>
      </button>
      <div class="gt-gallery-image-wrapper">
        <img id="gt-main-image" src="${getImageSrc(gtImages[0])}" alt="Dataset ground truth sample">
      </div>
      <button class="gt-nav gt-nav-next" aria-label="Next image">
        <i class="fas fa-chevron-right"></i>
      </button>
    </div>
    <div class="gt-gallery-info">
      <span class="gt-gallery-counter" id="gt-counter">1 / ${gtImages.length}</span>
    </div>
    <div class="gt-gallery-thumbnails" id="gt-thumbnails"></div>
  `;

  // Build thumbnails
  const thumbContainer = document.getElementById('gt-thumbnails');
  gtImages.forEach(function (img, i) {
    const thumb = document.createElement('div');
    thumb.className = 'gt-gallery-thumb' + (i === 0 ? ' active' : '');
    thumb.innerHTML = '<img src="' + getImageSrc(img) + '" alt="Sample ' + (i + 1) + '" loading="lazy">';
    thumb.addEventListener('click', function () { goTo(i); });
    thumbContainer.appendChild(thumb);
  });

  const mainImage = document.getElementById('gt-main-image');
  const counter = document.getElementById('gt-counter');
  const thumbs = thumbContainer.querySelectorAll('.gt-gallery-thumb');

  function goTo(index) {
    if (index < 0) index = gtImages.length - 1;
    if (index >= gtImages.length) index = 0;

    thumbs[currentIndex].classList.remove('active');
    currentIndex = index;

    mainImage.style.opacity = '0';
    var newSrc = getImageSrc(gtImages[currentIndex]);
    var preload = new Image();
    preload.onload = function () {
      mainImage.src = newSrc;
      mainImage.style.opacity = '1';
    };
    preload.src = newSrc;

    counter.textContent = (currentIndex + 1) + ' / ' + gtImages.length;
    thumbs[currentIndex].classList.add('active');
    thumbs[currentIndex].scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
  }

  // Prev / Next buttons
  gallery.querySelector('.gt-nav-prev').addEventListener('click', function () { goTo(currentIndex - 1); });
  gallery.querySelector('.gt-nav-next').addEventListener('click', function () { goTo(currentIndex + 1); });

  // Keyboard navigation (only when gallery section is in viewport)
  var galleryInView = false;
  var galleryObserver = new IntersectionObserver(function (entries) {
    galleryInView = entries[0].isIntersecting;
  }, { threshold: 0.2 });
  galleryObserver.observe(gallery.closest('.gt-gallery-section'));

  document.addEventListener('keydown', function (e) {
    if (!galleryInView) return;
    if (document.querySelector('.image-modal.active')) return;
    if (e.key === 'ArrowLeft') { goTo(currentIndex - 1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { goTo(currentIndex + 1); e.preventDefault(); }
  });

  // Swipe support for touch devices
  var touchStartX = 0;
  var imageWrapper = gallery.querySelector('.gt-gallery-image-wrapper');
  imageWrapper.addEventListener('touchstart', function (e) {
    touchStartX = e.changedTouches[0].screenX;
  }, { passive: true });
  imageWrapper.addEventListener('touchend', function (e) {
    var diff = e.changedTouches[0].screenX - touchStartX;
    if (Math.abs(diff) > 50) {
      if (diff > 0) goTo(currentIndex - 1);
      else goTo(currentIndex + 1);
    }
  });

  // Click main image to open in lightbox (reuse existing modal)
  mainImage.addEventListener('click', function () {
    var modal = document.querySelector('.image-modal');
    if (modal) {
      var modalImg = modal.querySelector('.img-modal-body img');
      modalImg.src = this.src;
      modalImg.alt = this.alt;
      modal.classList.add('active');
      document.body.style.overflow = 'hidden';
    }
  });
});

// Intersection Observer for scroll animations
document.addEventListener('DOMContentLoaded', function () {
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('fade-in-visible');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  // Add fade-in animation styles
  const style = document.createElement('style');
  style.textContent = `
    .fade-in {
      opacity: 0;
      transform: translateY(20px);
      transition: opacity 0.6s ease, transform 0.6s ease;
    }
    .fade-in-visible {
      opacity: 1;
      transform: translateY(0);
    }
  `;
  document.head.appendChild(style);

  // Apply to sections
  document.querySelectorAll('.contribution-card, .stat-card, .step-card, .example-pair').forEach(el => {
    el.classList.add('fade-in');
    observer.observe(el);
  });
});
