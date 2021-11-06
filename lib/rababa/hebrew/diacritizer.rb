# this refers to:
#   https://github.com/interscript/rababa/blob/main/python/diacritizer.py
# as well a drastic simplification of
#   https://github.com/almodhfer/Arabic_Diacritization/blob/master/config_manager.py

require "rababa/hebrew/encoders"
require "rababa/hebrew/nlp"
require "rababa/dataset"

module Rababa
  module Hebrew
    class Diacritizer
      # include Rababa::Reconciler

      def initialize(onnx_model_path, config)
        # load inference model from model_path
        @onnx_session = OnnxRuntime::InferenceSession.new(onnx_model_path)

        # load config
        @config = config
        @max_length = @config["max_len"]
        @batch_size = @config["batch_size"]

        # instantiate encoder's class
        @encoder = get_text_encoder
      end

      # preprocess text into indices
      def preprocess_text(text)
        # Cut text & warn user if text exceeds max_length
        if text.length > @max_length
          text = text[0..@max_length]
          warn("WARNING:: string has length > #{@max_length}:\n")
          warn("WARNING:: string cut at length: #{@max_length}\n")
          warn("text:: " + text)
        end

        # encode data with hebrew NLP (a la nakdimon)
        @encoder.encode_data(text)
      end

      # Diacritize single arabic strings:
      # string -> string
      def diacritize_text(text)
        # encode, indexing text data into Data class
        data = preprocess_text(text)

        ort_inputs = {
          "normalized" => [data.normalized +
            [0] * (@max_length - data.normalized.length)] * @batch_size
        }

        # onnx predictions
        vniqqud, vdagesh, vsin = predict_batch(ort_inputs)

        # decode text and prediction
        @encoder.decode_data(
          data.text, data.normalized,
          vdagesh[0], vsin[0], vniqqud[0]
        )
      end

      # download data from relative path and diacritize line by line
      # TODO: Rewrite this method based on diacritize_text
      def diacritize_file(path)
        # load data
        texts = File.open(path).map do |line|
          line.chomp.strip
        end

        # process batches
        out_texts = []
        idx = 0
        while idx + @batch_size <= texts.length
          # preprocess text
          vdata = texts[idx..idx + @batch_size - 1]
            .map.each { |t| preprocess_text(t) }

          # format for onnx
          ort_inputs = {
            "normalized" => vdata.map { |d|
                              d.normalized + [0] * (@max_length - d.normalized.length)
                            }
          }
          # perform onnx comput.
          vniqqud, vdagesh, vsin = predict_batch(ort_inputs)
          # decode data into string
          out_texts += (0..@batch_size - 1).map do |i|
            @encoder.decode_data(vdata[i].text, vdata[i].normalized,
              vdagesh[i], vsin[i], vniqqud[i])
          end
          idx += @batch_size
        end

        # process rest of data
        while idx < texts.length
          out_texts += [diacritize_text(texts[idx])]
          idx += 1
        end
        out_texts
      end

      # Call ONNX model with data transformed in batches
      def predict_batch(batch_data)
        # onnx predictions
        predicts = @onnx_session.run(nil, batch_data)

        # process dims: niqqud, dagesh, sin
        preds_niqqud = predicts[0].map do |p|
          p.map do |r|
            r.each_with_index.max[1]
          end
        end
        preds_dagesh = predicts[1].map do |p|
          p.map do |r|
            r.each_with_index.max[1]
          end
        end
        preds_sin = predicts[2].map do |p|
          p.map do |r|
            r.each_with_index.max[1]
          end
        end

        [preds_niqqud, preds_dagesh, preds_sin]
      end

      # Initialise text encoder from config params
      def get_text_encoder
        Encoders::TextEncoder.new
      end
    end
  end
end
