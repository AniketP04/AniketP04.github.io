window.MathJax = {
  tex: {
    tags: "ams",
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    displayMath: [
      ["$$", "$$"],
      ["\\[", "\\]"],
    ],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    renderActions: {
      addCss: [
        200,
        function (doc) {
          const style = document.createElement("style");
          style.innerHTML = `
          .mjx-container {
            color: inherit;
          }
        `;
          document.head.appendChild(style);
        },
        "",
      ],
    },
  },
  startup: {
    ready() {
      MathJax.startup.defaultReady();

      window.addEventListener("load", () => {
        MathJax.typesetPromise()
          .then(() => console.log("MathJax typeset complete"))
          .catch((err) => console.error("MathJax error:", err));
      });
    },
  },
};