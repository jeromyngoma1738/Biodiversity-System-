
setTimeout(function () {
    const alert = document.querySelector(".alert");
    if (alert) {
        alert.style.transition = "opacity 0.5s";
        alert.style.opacity = "0";

        setTimeout(() => {
            alert.remove();
        }, 500);
    }
}, 3000);


let slides = document.querySelectorAll(".slide");
let index = 0;

function showSlide(i){
    slides.forEach(slide => {
        slide.style.display = "none";
    });

    slides[i].style.display = "block";
}

document.querySelector(".next").onclick = function(){
    index++;

    if(index >= slides.length){
        index = 0;
    }

    showSlide(index);
};


document.querySelector(".prev").onclick = function(){
    index--;

    if(index < 0){
        index = slides.length - 1;
    }

    showSlide(index);
};


showSlide(index);

