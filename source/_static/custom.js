// Custom JavaScript for HRAG Documentation

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function () {

    // Add smooth scrolling to all internal links
    const internalLinks = document.querySelectorAll('a[href^="#"]');
    internalLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add copy functionality to code blocks
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        // Create copy button
        const copyButton = document.createElement('button');
        copyButton.className = 'copybutton';
        copyButton.textContent = '复制';
        copyButton.style.position = 'absolute';
        copyButton.style.top = '5px';
        copyButton.style.right = '5px';

        // Add click event
        copyButton.addEventListener('click', function () {
            const text = block.textContent;
            navigator.clipboard.writeText(text).then(() => {
                copyButton.textContent = '已复制!';
                setTimeout(() => {
                    copyButton.textContent = '复制';
                }, 2000);
            }).catch(err => {
                console.error('复制失败:', err);
                copyButton.textContent = '复制失败';
                setTimeout(() => {
                    copyButton.textContent = '复制';
                }, 2000);
            });
        });

        // Make parent relative positioned and add button
        const parent = block.parentElement;
        if (parent) {
            parent.style.position = 'relative';
            parent.appendChild(copyButton);
        }
    });

    // Add search functionality enhancement
    const searchInput = document.querySelector('input[type="search"]');
    if (searchInput) {
        searchInput.placeholder = '搜索 HRAG 文档...';

        // Add search suggestions
        searchInput.addEventListener('input', function () {
            const query = this.value.toLowerCase();
            if (query.length > 2) {
                // You can implement search suggestions here
                console.log('搜索查询:', query);
            }
        });
    }

    // Add table of contents highlighting
    const tocLinks = document.querySelectorAll('.wy-menu-vertical a');
    const sections = document.querySelectorAll('h1, h2, h3, h4, h5, h6');

    function highlightTOC() {
        const scrollPos = window.scrollY + 100;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;

            if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
                // Remove active class from all TOC links
                tocLinks.forEach(link => link.classList.remove('active'));

                // Add active class to corresponding TOC link
                const sectionId = section.id;
                const correspondingLink = document.querySelector(`.wy-menu-vertical a[href="#${sectionId}"]`);
                if (correspondingLink) {
                    correspondingLink.classList.add('active');
                }
            }
        });
    }

    // Throttle the scroll event
    let ticking = false;
    function updateTOC() {
        highlightTOC();
        ticking = false;
    }

    window.addEventListener('scroll', function () {
        if (!ticking) {
            requestAnimationFrame(updateTOC);
            ticking = true;
        }
    });

    // Add keyboard navigation
    document.addEventListener('keydown', function (e) {
        // Ctrl/Cmd + K for search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            if (searchInput) {
                searchInput.focus();
            }
        }

        // Escape to close search
        if (e.key === 'Escape') {
            if (searchInput && document.activeElement === searchInput) {
                searchInput.blur();
            }
        }
    });

    // Add performance monitoring
    if ('performance' in window) {
        window.addEventListener('load', function () {
            const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
            console.log(`页面加载时间: ${loadTime}ms`);
        });
    }

    // Add dark mode toggle (if needed)
    const darkModeToggle = document.createElement('button');
    darkModeToggle.textContent = '🌙';
    darkModeToggle.className = 'dark-mode-toggle';
    darkModeToggle.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        background: #2980B9;
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        cursor: pointer;
        font-size: 16px;
    `;

    darkModeToggle.addEventListener('click', function () {
        document.body.classList.toggle('dark-mode');
        this.textContent = document.body.classList.contains('dark-mode') ? '☀️' : '🌙';
    });

    document.body.appendChild(darkModeToggle);

    // Add code syntax highlighting enhancement
    const codeElements = document.querySelectorAll('code');
    codeElements.forEach(code => {
        // Add language detection
        const className = code.className;
        if (className && className.includes('language-')) {
            const language = className.replace('language-', '');
            code.setAttribute('data-language', language);
        }
    });

    // Add responsive navigation enhancement
    const mobileMenuToggle = document.querySelector('.wy-breadcrumbs-aside .fa-bars');
    if (mobileMenuToggle) {
        mobileMenuToggle.addEventListener('click', function () {
            const nav = document.querySelector('.wy-nav-side');
            if (nav) {
                nav.classList.toggle('nav-open');
            }
        });
    }

    // Add anchor links to headings
    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    headings.forEach(heading => {
        if (heading.id) {
            const anchor = document.createElement('a');
            anchor.href = `#${heading.id}`;
            anchor.className = 'headerlink';
            anchor.innerHTML = '¶';
            anchor.style.cssText = `
                margin-left: 10px;
                text-decoration: none;
                color: #ccc;
                font-size: 0.8em;
                opacity: 0;
                transition: opacity 0.2s;
            `;

            heading.appendChild(anchor);

            // Show anchor on hover
            heading.addEventListener('mouseenter', function () {
                anchor.style.opacity = '1';
            });

            heading.addEventListener('mouseleave', function () {
                anchor.style.opacity = '0';
            });
        }
    });

    // Add progress bar
    const progressBar = document.createElement('div');
    progressBar.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 0%;
        height: 3px;
        background: #2980B9;
        z-index: 1001;
        transition: width 0.1s ease;
    `;
    document.body.appendChild(progressBar);

    function updateProgressBar() {
        const scrollTop = window.scrollY;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const scrollPercent = (scrollTop / docHeight) * 100;
        progressBar.style.width = scrollPercent + '%';
    }

    window.addEventListener('scroll', updateProgressBar);

    // Add print styles
    const printStyles = document.createElement('style');
    printStyles.textContent = `
        @media print {
            .wy-nav-side, .dark-mode-toggle, .copybutton {
                display: none !important;
            }
            .wy-nav-content {
                margin-left: 0 !important;
            }
            pre {
                white-space: pre-wrap !important;
                word-wrap: break-word !important;
            }
        }
    `;
    document.head.appendChild(printStyles);

    console.log('HRAG 文档增强功能已加载');
}); 