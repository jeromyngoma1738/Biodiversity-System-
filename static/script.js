//show function used to display the specific form based on the ID provided.
//formId parameter used to capture the ID of the the form to be displayed
//document.querySelectorALL selects all HTML element with a form box class and the FOREACH to iterate to remove class that active
//document.getElementBYid () get the element based on the id provided
//Class.add () gets the active to the  class selected

function showform(formId){
    document.querySelectorAll(".form-box").forEach(form => form.classList.remove("active")); 
    document.getElementById(formId).classList.add("active");

}