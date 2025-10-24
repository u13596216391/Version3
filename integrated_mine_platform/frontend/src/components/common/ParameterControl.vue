<template>
  <div class="parameter-control">
    <div class="flex justify-between items-center mb-2">
      <label class="text-sm font-semibold text-gray-300">{{ label }}</label>
      <span class="text-lg font-bold text-violet-400">{{ modelValue }}</span>
    </div>
    <input 
      type="range" 
      :value="modelValue" 
      @input="$emit('update:modelValue', parseFloat($event.target.value))"
      :min="min" 
      :max="max" 
      :step="step"
      class="slider w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
    />
    <p class="text-xs text-gray-500 mt-1">{{ description }}</p>
  </div>
</template>

<script setup>
defineProps({
  label: String,
  description: String,
  modelValue: [Number, String],
  min: {
    type: Number,
    default: 0
  },
  max: {
    type: Number,
    default: 100
  },
  step: {
    type: Number,
    default: 1
  }
})

defineEmits(['update:modelValue'])
</script>

<style scoped>
.slider::-webkit-slider-thumb {
  appearance: none;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  cursor: pointer;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.slider::-moz-range-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  cursor: pointer;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  border: none;
}

.slider:focus {
  outline: none;
}

.slider:focus::-webkit-slider-thumb {
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
}
</style>
