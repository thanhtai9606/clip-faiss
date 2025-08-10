document.getElementById('searchButton')
.addEventListener('click', function() {
  const searchTerm = document.getElementById('searchInput').value;
  if (searchTerm.trim() !== '') {
    const apiUrl = `/search?search_query=${encodeURIComponent(searchTerm)}`;
    fetch(apiUrl)
      .then(response => response.json())
      .then(data => {
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = '';
        data.results.forEach(item => {
          const wrap = document.createElement('div');
          wrap.className = 'thumb';

          const img = document.createElement('img');
          img.src = item.path;
          img.alt = 'result';

          const score = document.createElement('span');
          score.className = 'score';
          score.textContent = `score: ${Number(item.score * 100).toFixed(3)}%`;

          wrap.appendChild(img);
          wrap.appendChild(score);
          resultsDiv.appendChild(wrap);
        });
      });
  }
});
