(function () {
  var stage = document.getElementById("demo-stage");
  var button = document.getElementById("demo-load");
  var poster = document.getElementById("demo-poster");
  if (!stage || !button || !poster) {
    return;
  }

  var gifSrc = "assets/result.gif";
  var loaded = false;

  function revealGif() {
    if (loaded) {
      return;
    }
    loaded = true;

    var img = document.createElement("img");
    img.width = 426;
    img.height = 240;
    img.alt =
      "仓库根目录 result.gif：手持拍摄的 Android 应用预览，镜头对着显示器上的文字页面，预览上叠有检测框。";
    img.decoding = "async";
    img.loading = "lazy";
    img.src = gifSrc;

    img.addEventListener("error", function () {
      loaded = false;
      var note = document.getElementById("demo-error");
      if (note) {
        note.hidden = false;
      }
      button.hidden = false;
      button.disabled = false;
      button.textContent = "未能加载动画，请改用直接链接";
    });

    poster.replaceWith(img);
    stage.classList.add("is-playing");
    button.hidden = true;
  }

  button.addEventListener("click", revealGif);
})();
