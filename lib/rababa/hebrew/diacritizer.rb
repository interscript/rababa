# this refers to:
#   https://github.com/interscript/rababa/blob/main/python/diacritizer.py
# as well a drastic simplification of
#   https://github.com/almodhfer/Arabic_Diacritization/blob/master/config_manager.py

require "rababa/hebrew/encoders"
require "rababa/hebrew/nlp"
require "rababa/hebrew/dataset"
require "rababa/diacritizer"

module Rababa
  module Hebrew
    class Diacritizer < Rababa::Diacritizer
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

      # Diacritize single arabic strings or a batch:
      # string -> string
      # array -> array
      def diacritize_text(text)
        if text.is_a?(String)
          text = [text.strip] * @batch_size
          seq = [preprocess_text(text.first)] * @batch_size
          return_single = true
        else
          seq = text.map { |i| preprocess_text(i) }
        end

        ort_inputs = {
          "normalized" => seq.map { |d|
            d.normalized + [0] * (@max_length - d.normalized.length)
          }
        }

        # onnx predictions
        vniqqud, vdagesh, vsin = predict_batch(ort_inputs)

        size = return_single ? 1 : @batch_size

        out = size.times.map do |i|
          @encoder.decode_data(seq[i].text, seq[i].normalized,
            vdagesh[i], vsin[i], vniqqud[i])
        end

        return_single ? out.first : out
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
