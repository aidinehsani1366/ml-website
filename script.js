$(document).ready(function () {
    $('input[type="file"]').change(function () {
        if (this.files && this.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#uploaded-image').attr('src', e.target.result);
            };
            reader.readAsDataURL(this.files[0]);
        }
    });
});
