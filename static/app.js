const resultsDiv = document.getElementById('results');

    function renderResults(list) {
      resultsDiv.innerHTML = '';
      if (!Array.isArray(list) || list.length === 0) {
        resultsDiv.innerHTML = '<div class="muted">Không có kết quả.</div>';
        return;
      }
      list.forEach(item => {
        const wrap = document.createElement('div');
        wrap.className = 'thumb';
        wrap.title = `score: ${Number(item.score).toFixed(3)}`; // tooltip nhanh

        const img = document.createElement('img');
        img.src = item.path;
        img.alt = 'result';

        const score = document.createElement('span');
        score.className = 'score';
        score.textContent = `score: ${Number(item.score).toFixed(3)}`;

        wrap.appendChild(img);
        wrap.appendChild(score);
        resultsDiv.appendChild(wrap);
      });
    }

    // TEXT SEARCH
    document.getElementById('searchButton')
      .addEventListener('click', function() {
        const searchTerm = document.getElementById('searchInput').value;
        if (searchTerm.trim() !== '') {
          const apiUrl = `/search?search_query=${encodeURIComponent(searchTerm)}&k=12`;
          fetch(apiUrl)
            .then(r => r.json())
            .then(data => renderResults(data.results))
            .catch(() => { resultsDiv.innerHTML = '<div class="muted">Lỗi gọi API /search</div>'; });
        }
      });

    // IMAGE SEARCH
    const fileInput = document.getElementById('fileInput');
    const imgForm   = document.getElementById('imgForm');
    const imgButton = document.getElementById('imgButton');
    const preview   = document.getElementById('preview');
    const previewImg= document.getElementById('previewImg');
    const previewName= document.getElementById('previewName');

    // Preview ảnh đã chọn
    fileInput.addEventListener('change', () => {
      const f = fileInput.files && fileInput.files[0];
      if (!f) { preview.style.display = 'none'; return; }
      const url = URL.createObjectURL(f);
      previewImg.src = url;
      previewName.textContent = f.name;
      preview.style.display = 'flex';
    });

    imgForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const f = fileInput.files && fileInput.files[0];
      if (!f) return;

      imgButton.disabled = true;
      const fd = new FormData();
      fd.append('file', f);

      fetch('/search-image?k=12', {
        method: 'POST',
        body: fd
      })
        .then(r => r.json())
        .then(data => renderResults(data.results))
        .catch(() => { resultsDiv.innerHTML = '<div class="muted">Lỗi gọi API /search-image</div>'; })
        .finally(() => { imgButton.disabled = false; });
    });