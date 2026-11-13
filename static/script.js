console.log("CareerFlix AI loaded successfully!");

document.addEventListener("DOMContentLoaded", function () {

    const interviewButton =
        document.querySelector(".primary-btn");

    const resumeButton =
        document.querySelector(".secondary-btn");


    if (interviewButton) {

        interviewButton.addEventListener(
            "click",
            function () {

                alert(
                    "AI Mock Interview module coming next!"
                );

            }
        );
    }


    if (resumeButton) {

        resumeButton.addEventListener(
            "click",
            function () {

                alert(
                    "AI Resume Analyzer module coming next!"
                );

            }
        );
    }

});