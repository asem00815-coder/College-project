fetch(`assets/RusMap2.svg`)
   .then(res => res.text())
   .then(svg => {
      document.getElementById(`map`).innerHTML = svg;
   })
.catch(err => console.error(err));