document.getElementById('add-course').addEventListener('click', function() {
    const courseContainer = document.getElementById('course-container');
    const courseCount = courseContainer.children.length + 1;
    const newCourseRow = document.createElement('div');
    newCourseRow.className = 'course-row';
    newCourseRow.innerHTML = `
        <label>Course ${courseCount}</label>
        <select class="grade">
            <option value="0-50">F</option>
            <option value="51-55">C</option>
            <option value="56-60">B</option>
            <option value="61-70">B+</option>
            <option value="71-80">A</option>
            <option value="81-90">A+</option>
            <option value="91-100">O</option>
        </select> 
        <label>Credits</label>
        <select class="credit">
            <option value="0">0</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
            <option value="5">5</option>
        </select>
    `;
    courseContainer.appendChild(newCourseRow);
});

document.getElementById('add-certificate').addEventListener('click', function() {
    const certificateContainer = document.getElementById('certificate-container');
    const certificateCount = certificateContainer.children.length + 1;
    
    // Limit the number of certificates that can be added (optional)
    if (certificateCount > 5) {
        alert('Maximum of 5 certificates allowed');
        return;
    }
    
    const newCertificateRow = document.createElement('div');
    newCertificateRow.className = 'Certificate-row'; // Note: Case-sensitive, must match CSS
    newCertificateRow.innerHTML = `
        <label>Certificate-${certificateCount}</label>
        <select class="certificate-category">
            <option value="5">CISCO, CCNA, CCNP, MCNA, MCNP, Matlab, Redhat, IBM</option>
            <option value="3">NPTL</option>
            <option value="2">Coursera</option>
            <option value="1">Programming Certifications (C, C++, Java, Python, etc)</option>
            <option value="0.5">Udemy</option>
            <option value="0">Null</option>
        </select>
    `;
    certificateContainer.appendChild(newCertificateRow);
});

// Add event listener for the calculate button
document.getElementById('calculate').addEventListener('click', async function() {
    // Collect data from form
    const courseRows = document.querySelectorAll('.course-row');
    const courses = [];
    
    courseRows.forEach(row => {
        const gradeSelect = row.querySelector('.grade');
        const creditSelect = row.querySelector('.credit');
        
        // Add null check
        if (!gradeSelect || !creditSelect) {
            console.error('Could not find grade or credit select in row:', row);
            return;
        }
        
        courses.push({
            grade: gradeSelect.value,
            credits: parseInt(creditSelect.value)
        });
    });
    
    // Get attendance and CGPA
    const attendanceSelect = document.getElementById('attendance-select');
    const cgpaSelect = document.getElementById('cgpa-select');
    
    // Get internship value
    const internshipSelect = document.getElementById('Internship-select');
    
    // Get certificates
    const certificateRows = document.querySelectorAll('.Certificate-row');
    const certificates = [];
    
    certificateRows.forEach(row => {
        const certSelect = row.querySelector('.certificate-category');
        if (certSelect) {
            certificates.push(certSelect.value);
        }
    });
    
    // Add null check
    if (!attendanceSelect || !cgpaSelect) {
        console.error('Could not find attendance or CGPA select elements');
        alert('Error: Form elements missing. Please refresh the page.');
        return;
    }
    
    const attendance = attendanceSelect.value;
    const cgpa = cgpaSelect.value;
    
    // Prepare internship value
    let internship = "0";
    if (internshipSelect) {
        internship = internshipSelect.value;
    }
    
    // Log data for debugging
    console.log('Sending data to backend:', {
        courses: courses,
        attendance: attendance,
        cgpa: cgpa,
        internship: internship,
        certificates: certificates
    });
    
    // Prepare data for submission
    const studentData = {
        courses: courses,
        attendance: attendance,
        cgpa: cgpa,
        internship: internship,
        certificates: certificates
    };
    
    try {
        // Show loading state
        document.getElementById('calculate').textContent = 'Processing...';
        document.getElementById('calculate').disabled = true;
        
        // Add debug logging for request
        console.log('Sending request to /predict endpoint');
        
        // Send data to backend for AI prediction
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(studentData)
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            // Try to get the error message from the response
            const errorData = await response.json().catch(() => null);
            const errorMessage = errorData && errorData.error 
                ? errorData.error 
                : `Network response was not ok: ${response.status}`;
            throw new Error(errorMessage);
        }
        
        const predictionResult = await response.json();
        console.log('Prediction result:', predictionResult);
        
        // Display the results
        displayResults(predictionResult);
    } catch (error) {
        console.error('Error in prediction request:', error);
        // Provide more detailed error information
        const errorMessage = document.getElementById('results-container');
        errorMessage.innerHTML = `
            <div class="error-message">
                <h3>Error Getting Prediction</h3>
                <p>There was a problem connecting to the prediction service:</p>
                <p><code>${error.message}</code></p>
                <p>Make sure the Flask server is running at the correct address.</p>
            </div>
        `;
        errorMessage.style.display = 'block';
        alert('Failed to get prediction. Please check the browser console for more details.');
    } finally {
        document.getElementById('calculate').textContent = 'My Performance';
        document.getElementById('calculate').disabled = false;
    }
});

// Function to display prediction results
function displayResults(results) {
    const resultsContainer = document.getElementById('results-container');
    
    // Create content for results
    resultsContainer.innerHTML = `
        <i><h2>Prediction Results</h2></i>
        <div class="results-content">
            <p><strong>Predicted Final Grade Range:</strong> ${results.predictedGrade}</p>
            <p><strong>Performance Category:</strong> ${results.performanceCategory}</p>
            <p><strong>Risk Level:</strong> ${results.riskLevel}</p>
            <div class="recommendation">
                <h3>Recommendations:</h3>
                <ul>
                    ${results.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
        </div>
    `;
    
    // Make sure the results container is visible
    resultsContainer.style.display = 'block';
}