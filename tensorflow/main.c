#include <stdio.h>
#include <tensorflow/lite/c/c_api.h>

int main() {
	const char* model_path = "model.tflite";
	TfLiteModel* model = TfLiteModelCreateFromFile(model_path);
	if (!model) { printf("\n"); return -1; }

	TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
	TfLiteInterpreterOptionsDelete(options);
	TfLiteModelDelete(model);
	if (!interpreter) { printf("\n"); return -1; }

	if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
		printf("\n"); TfLiteInterpreterDelete(interpreter); return -1;
	}

	TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
	float input_data[1] = {1.0f};
	TfLiteTensorCopyFromBuffer(input_tensor, input_data, sizeof(input_data));

	if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
		printf("\n"); TfLiteInterpreterDelete(interpreter); return -1;
	}

	const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
	float output_data[1] = {0};
	TfLiteTensorCopyToBuffer(output_tensor, output_data, sizeof(output_data));
	printf(": %f\n", output_data[0]);

	TfLiteInterpreterDelete(interpreter);
	return 0;
}

