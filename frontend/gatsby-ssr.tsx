import React from "react";

const codeToRunOnClient = `(function() {
  try {
    var mode = localStorage.getItem('darkmode');
    document.getElementsByTagName("html")[0].className === 'dark' ? 'dark' : 'light';
  } catch (e) {}
})();`;

export const onRenderBody = ({ setHeadComponents }) =>
  setHeadComponents([
    // Preconnect for Inter font (Google Fonts CDN)
    <link key="gfonts-preconnect" rel="preconnect" href="https://fonts.googleapis.com" />,
    <link key="gfonts-preconnect2" rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />,
    <link
      key="inter-font"
      rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"
    />,
    <script
      key="dark-mode-script"
      dangerouslySetInnerHTML={{ __html: codeToRunOnClient }}
    />,
  ]);
