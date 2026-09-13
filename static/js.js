
//AUTO-HIDE FLASK ALERT

setTimeout(function () {
const alert = document.querySelector(".alert");
if (alert) {
    alert.style.transition = "opacity 0.5s";
    alert.style.opacity = "0";
    setTimeout(function () {
        alert.remove();
    }, 500);
}
}, 3000);

/* ==============================
WILDLIFE GALLERY SLIDER
============================== */

const slides = document.querySelectorAll(".slide");
const nextButton = document.querySelector(".next");
const prevButton = document.querySelector(".prev");

let index = 0;
/* Show selected slide */
function showSlide(i) {
if (slides.length === 0) {
    return;
}

slides.forEach(function (slide) {slide.style.display = "none";});

slides[i].style.display = "block";}

/* Next button */

if (nextButton) {
nextButton.addEventListener("click", function () {
    index++;
    if (index >= slides.length) {
        index = 0;
    }
    showSlide(index);
});}

/* Previous button */
if (prevButton) {
prevButton.addEventListener("click", function () {
    index--;
    if (index < 0) {
        index = slides.length - 1;
    }
    showSlide(index);
});
}

/* Show first slide */

showSlide(index);
