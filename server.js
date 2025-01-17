const express = require('express');
const ort = require('onnxruntime-node');
const path = require('path');
const { Pool } = require('pg'); // Import PostgreSQL library

// Initialize the Express app
const app = express();
const port = 3000;

// Middleware to parse JSON requests
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// PostgreSQL connection configuration
const pool = new Pool({
    user: 'rashad', // Replace with your PostgreSQL username
    host: '18.143.206.136',           // Replace with your database host (e.g., EC2 public IP for remote)
    database: 'water_potability', // Replace with your database name
    password: 'rashad123', // Replace with your PostgreSQL password
    port: 5432,                  // Default PostgreSQL port
});

// Load the ONNX model
const modelPath = path.join(__dirname, 'water_potability_model2.onnx');
let session;

(async () => {
    try {
        session = await ort.InferenceSession.create(modelPath);
        console.log('ONNX model loaded successfully.');
    } catch (error) {
        console.error('Error loading ONNX model:', error);
    }
})();

// Prediction endpoint
app.post('/predict', async (req, res) => {
    try {
        // Extract input data from the request body
        const inputData = req.body;

        // Log the input data for debugging
        console.log('Input Data:', inputData);

        // Convert input data to the required format
        const inputValues = [
            parseFloat(inputData.ph),
            parseFloat(inputData.Hardness),
            parseFloat(inputData.Solids),
            parseFloat(inputData.Chloramines),
            parseFloat(inputData.Sulfate),
            parseFloat(inputData.Conductivity),
            parseFloat(inputData.Organic_carbon),
            parseFloat(inputData.Trihalomethanes),
            parseFloat(inputData.Turbidity),
        ];

        // Log the converted input values for debugging
        console.log('Converted Input Values:', inputValues);

        // Create the input tensor
        const inputTensor = new ort.Tensor('float32', Float32Array.from(inputValues), [1, 9]);

        // Log the input tensor for debugging
        console.log('Input Tensor:', inputTensor);

        // Ensure the session is loaded
        if (!session) {
            throw new Error('ONNX session is not loaded.');
        }

        // Run the model
        const feeds = { float_input: inputTensor }; // Use the correct input name: "float_input"
        console.log('Feeds:', feeds); // Debugging: Log the feeds

        const results = await session.run(feeds);
        console.log('Results:', results); // Debugging: Log the results

        // Get the probabilities
        const probabilities = results.probabilities.data; // Raw probabilities for each class
        console.log('Probabilities:', probabilities);

        // Determine the predicted class based on probabilities
        const prediction = probabilities[1] > probabilities[0] ? 1 : 0; // Compare probabilities for class 1 and class 0

        // Save the prediction to the database
        const query = `
            INSERT INTO predictions (ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity, potability)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING *;
        `;
        const values = [
            inputData.ph,
            inputData.Hardness,
            inputData.Solids,
            inputData.Chloramines,
            inputData.Sulfate,
            inputData.Conductivity,
            inputData.Organic_carbon,
            inputData.Trihalomethanes,
            inputData.Turbidity,
            prediction === 1, // potability (true if safe, false if not safe)
        ];

        const dbResult = await pool.query(query, values);
        console.log('Prediction saved to database:', dbResult.rows[0]);

        // Send the response
        if (prediction === 1) {
            res.json({ 
                message: "The water is safe to drink",
                probabilities: Array.from(probabilities) // Convert Float32Array to a regular array
            });
        } else {
            res.json({ 
                message: "The water is not safe to drink",
                probabilities: Array.from(probabilities) // Convert Float32Array to a regular array
            });
        }
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: error.message }); // Send the actual error message
    }
});

// Endpoint to fetch prediction history
app.get('/history', async (req, res) => {
    try {
        const query = 'SELECT * FROM predictions ORDER BY prediction_time DESC;';
        const result = await pool.query(query);
        res.json(result.rows); // Send the history as JSON
    } catch (error) {
        console.error('Error fetching history:', error);
        res.status(500).json({ error: 'Database error' });
    }
});

// Route to serve the frontend
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start the server
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});