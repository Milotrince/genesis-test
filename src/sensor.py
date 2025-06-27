import json
import time
import threading
from collections import deque
from typing import Optional, Dict, Any, List, Union
import os
from datetime import datetime
import numpy as np




def get_pinned_vertex_indices(original_verts, pinned_verts, tolerance=1e-6):
    """
    Get the indices of pinned vertices based on their positions.
    Note: trimesh is not guaranteed to preserve vertex order.
    
    Args:
        original_verts: Array of vertex positions from the original mesh
        pinned_verts: Array of pinned vertex positions 
        tolerance: Tolerance for floating-point comparison (default: 1e-6)
    """
    pinned_indices = []
    
    # Convert to numpy arrays if they aren't already
    original_verts = np.asarray(original_verts)
    pinned_verts = np.asarray(pinned_verts)
    
    for i, coord in enumerate(original_verts):
        print(f"Checking vertex {i}: {coord}")
        
        # Check distance to all pinned vertices
        distances = np.linalg.norm(pinned_verts - coord, axis=1)
        closest_matches = np.where(distances < tolerance)[0]
        
        if len(closest_matches) > 0:
            closest_idx = closest_matches[0]
            print(f"Vert {i} matches pinned vertex {closest_idx} with distance {distances[closest_idx]:.2e}")
            pinned_indices.append(i)
    
    return pinned_indices



# ------

class TactileForceSensor:
    """Sensor for tactile force measurement."""

    def __init__(self, frequency: float):
        super().__init__(frequency)


class DataStreamer:
    def __init__(
        self,
        filename: Optional[str] = None,
        buffer_size: int = 1000,
        chunk_size: int = 100,
        auto_flush_interval: float = 5.0,
        output: str = "sensor_data",
        output_format: str = "txt",
    ):
        """
        Args:
            filename: Optional filename. If None, generates timestamp-based name.
            buffer_size: Maximum number of readings to keep in memory
            chunk_size: Number of readings to write at once
            auto_flush_interval: Seconds between automatic flushes
            output: Directory to store sensor data files or base path for output
            output_format: Output format - "jsonl", "txt", or "npz" (used if no extension in filename)
        """
        self.sensor_data = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.auto_flush_interval = auto_flush_interval
        self.output = output
        self.output_format = output_format.lower()

        # File streaming state
        self.current_file = None
        self.current_filename = None
        self.current_format = None
        self.pending_writes = []
        self.total_readings = 0

        # For NPZ format, collect all data in memory
        self.npz_data_buffer = []

        # Threading for auto-flush
        self._flush_timer = None
        self._lock = threading.Lock()
        self._streaming_enabled = False

        # Ensure output directory exists
        os.makedirs(output, exist_ok=True)
        
        # Start streaming immediately
        self._start_streaming(filename)

    def _detect_format_from_filename(self, filename: str) -> str:
        """Detect output format from filename extension."""
        if filename.endswith('.npz'):
            return 'npz'
        elif filename.endswith('.txt'):
            return 'txt'
        elif filename.endswith('.jsonl') or filename.endswith('.json'):
            return 'jsonl'
        else:
            return self.output_format

    def _start_streaming(self, filename: Optional[str] = None):
        """
        Start streaming sensor data to a file.

        Args:
            filename: Optional custom filename. If None, generates timestamp-based name.
        """
        with self._lock:
            if self._streaming_enabled:
                self.stop_streaming()

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if self.output_format == "txt":
                    filename = f"sensor_data_{timestamp}.txt"
                elif self.output_format == "npz":
                    filename = f"sensor_data_{timestamp}.npz"
                else:
                    filename = f"sensor_data_{timestamp}.jsonl"
            
            # Detect format from filename extension or use default
            format_to_use = self._detect_format_from_filename(filename)

            self.current_filename = os.path.join(self.output, filename)
            self.current_format = format_to_use

            # For NPZ, we don't open file until the end
            if format_to_use != "npz":
                self.current_file = open(self.current_filename, 'w', encoding='utf-8')

            self._streaming_enabled = True
            self.pending_writes.clear()
            self.npz_data_buffer.clear()

            # Start auto-flush timer (not needed for NPZ)
            if format_to_use != "npz":
                self._start_auto_flush()

            print(
                f"Started streaming to: {self.current_filename} (format: {format_to_use})"
            )
    
    def start_streaming(
        self, filename: Optional[str] = None, output_format: Optional[str] = None
    ) -> str:
        """
        Start streaming sensor data to a file.

        Args:
            filename: Optional custom filename. If None, generates timestamp-based name.
            output_format: Override the default output format for this session

        Returns:
            The filename being used for streaming
        """
        with self._lock:
            if self._streaming_enabled:
                self.stop_streaming()

            # Use provided format or default
            format_to_use = (
                output_format.lower() if output_format else self.output_format
            )

            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if format_to_use == "txt":
                    filename = f"sensor_data_{timestamp}.txt"
                elif format_to_use == "npz":
                    filename = f"sensor_data_{timestamp}.npz"
                else:
                    filename = f"sensor_data_{timestamp}.jsonl"
            else:
                # If filename is provided, detect format from extension
                detected_format = self._detect_format_from_filename(filename)
                if output_format is None:
                    format_to_use = detected_format

            self.current_filename = os.path.join(self.output, filename)
            self.current_format = format_to_use

            # For NPZ, we don't open file until the end
            if format_to_use != "npz":
                self.current_file = open(self.current_filename, "w", encoding='utf-8')

            self._streaming_enabled = True
            self.pending_writes.clear()
            self.npz_data_buffer.clear()

            # Start auto-flush timer (not needed for NPZ)
            if format_to_use != "npz":
                self._start_auto_flush()

            print(
                f"Started streaming to: {self.current_filename} (format: {format_to_use})"
            )
            return self.current_filename

    def stop_streaming(self):
        """Stop streaming and flush any remaining data."""
        with self._lock:
            if not self._streaming_enabled:
                return

            # Cancel auto-flush timer
            if self._flush_timer:
                self._flush_timer.cancel()
                self._flush_timer = None

            # Handle different formats
            if self.current_format == "npz":
                self._save_npz_data()
            else:
                # Flush remaining data for text formats
                self._flush_pending_writes()

                # Close file
                if self.current_file:
                    self.current_file.close()
                    self.current_file = None

            self._streaming_enabled = False
            print(f"Stopped streaming. Total readings written: {self.total_readings}")
            print(f"File saved: {self.current_filename}")

    def _save_npz_data(self):
        """Save all buffered data to NPZ format."""
        if not self.npz_data_buffer:
            return

        try:
            # Convert data to numpy arrays
            timestamps = np.array([d["timestamp"] for d in self.npz_data_buffer])
            sensor_types = np.array([d["sensor_type"] for d in self.npz_data_buffer])

            # Handle different value types
            values = []
            metadata_list = []

            for d in self.npz_data_buffer:
                values.append(d["value"])
                metadata_list.append(json.dumps(d.get("metadata", {})))

            # Try to convert values to numpy array (works if all values are numeric)
            try:
                values_array = np.array(values)
            except:
                # If values are mixed types, keep as object array
                values_array = np.array(values, dtype=object)

            metadata_array = np.array(metadata_list)

            # Save to NPZ file
            np.savez_compressed(
                self.current_filename,
                timestamps=timestamps,
                sensor_types=sensor_types,
                values=values_array,
                metadata=metadata_array,
            )

            self.total_readings = len(self.npz_data_buffer)

        except Exception as e:
            print(f"Error saving NPZ file: {e}")

    def _start_auto_flush(self):
        """Start the auto-flush timer."""
        if self._flush_timer:
            self._flush_timer.cancel()

        self._flush_timer = threading.Timer(self.auto_flush_interval, self._auto_flush)
        self._flush_timer.start()

    def _auto_flush(self):
        """Automatically flush pending writes."""
        with self._lock:
            if self._streaming_enabled and self.current_format != "npz":
                self._flush_pending_writes()
                self._start_auto_flush()

    def _flush_pending_writes(self):
        """Write pending data to file."""
        if not self.current_file or not self.pending_writes:
            return

        try:
            for data in self.pending_writes:
                if self.current_format == "txt":
                    # Plain text format: one data point per line
                    line = f"{data['timestamp']:.6f},{data['sensor_type']},{data['value']}\n"
                    self.current_file.write(line)
                else:
                    # JSONL format
                    json_line = json.dumps(data) + "\n"
                    self.current_file.write(json_line)

            self.current_file.flush()
            self.total_readings += len(self.pending_writes)
            self.pending_writes.clear()

        except Exception as e:
            print(f"Error writing to file: {e}")

    def add_sensor_reading(
        self,
        sensor_type: str,
        value: Union[float, int, Dict],
        timestamp: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Add a sensor reading with automatic streaming support.

        Args:
            sensor_type: Type of sensor (e.g., 'temperature', 'humidity')
            value: Sensor reading value
            timestamp: Optional timestamp (uses current time if None)
            metadata: Optional additional metadata
        """
        if timestamp is None:
            timestamp = time.time()

        reading = {
            "sensor_type": sensor_type,
            "value": value,
            "timestamp": timestamp,
            "metadata": metadata or {},
        }

        with self._lock:
            # Add to in-memory buffer
            self.sensor_data.append(reading)

            # Add to appropriate buffer if streaming
            if self._streaming_enabled:
                if self.current_format == "npz":
                    self.npz_data_buffer.append(reading)
                else:
                    self.pending_writes.append(reading)

                    # Check if we need to flush (not for NPZ)
                    if len(self.pending_writes) >= self.chunk_size:
                        self._flush_pending_writes()

    def get_recent_readings(
        self, sensor_type: Optional[str] = None, count: Optional[int] = None
    ) -> List[Dict]:
        """
        Get recent sensor readings from memory buffer.

        Args:
            sensor_type: Filter by sensor type (optional)
            count: Maximum number of readings to return (optional)

        Returns:
            List of sensor readings
        """
        with self._lock:
            readings = list(self.sensor_data)

        if sensor_type:
            readings = [r for r in readings if r["sensor_type"] == sensor_type]

        if count:
            readings = readings[-count:]

        return readings

    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get statistics about the streaming operation."""
        with self._lock:
            stats = {
                "streaming_enabled": self._streaming_enabled,
                "current_file": self.current_filename,
                "output_format": getattr(self, "current_format", self.output_format),
                "total_readings_written": self.total_readings,
                "buffer_usage": len(self.sensor_data),
                "buffer_size": self.buffer_size,
                "chunk_size": self.chunk_size,
            }

            if hasattr(self, "current_format") and self.current_format == "npz":
                stats["npz_buffer_size"] = len(self.npz_data_buffer)
            else:
                stats["pending_writes"] = len(self.pending_writes)

            return stats

    def force_flush(self):
        """Force flush any pending writes to file."""
        with self._lock:
            if self._streaming_enabled and self.current_format != "npz":
                self._flush_pending_writes()

    def read_sensor_file(self, filename: str) -> List[Dict]:
        """
        Read sensor data from a previously saved file.

        Args:
            filename: Name of the file to read

        Returns:
            List of sensor readings
        """
        filepath = os.path.join(self.output, filename)
        readings = []

        try:
            if filename.endswith(".npz"):
                # Read NPZ file
                data = np.load(filepath)
                for i in range(len(data["timestamps"])):
                    reading = {
                        "timestamp": float(data["timestamps"][i]),
                        "sensor_type": str(data["sensor_types"][i]),
                        "value": (
                            data["values"][i].item()
                            if hasattr(data["values"][i], "item")
                            else data["values"][i]
                        ),
                        "metadata": (
                            json.loads(str(data["metadata"][i]))
                            if len(data["metadata"]) > i
                            else {}
                        ),
                    }
                    readings.append(reading)
            elif filename.endswith(".txt"):
                # Read plain text file
                with open(filepath, "r") as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split(",", 2)
                            if len(parts) >= 3:
                                reading = {
                                    "timestamp": float(parts[0]),
                                    "sensor_type": parts[1],
                                    "value": (
                                        float(parts[2])
                                        if parts[2]
                                        .replace(".", "")
                                        .replace("-", "")
                                        .isdigit()
                                        else parts[2]
                                    ),
                                    "metadata": {},
                                }
                                readings.append(reading)
            else:
                # Read JSONL file
                with open(filepath, "r") as f:
                    for line in f:
                        if line.strip():
                            readings.append(json.loads(line))
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")

        return readings

    def list_sensor_files(self) -> List[str]:
        """List all sensor data files in the output directory."""
        try:
            files = [
                f
                for f in os.listdir(self.output)
                if f.endswith(".jsonl")
                or f.endswith(".json")
                or f.endswith(".txt")
                or f.endswith(".npz")
            ]
            return sorted(files)
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def cleanup_old_files(self, keep_latest: int = 10):
        """
        Remove old sensor data files, keeping only the latest ones.

        Args:
            keep_latest: Number of most recent files to keep
        """
        files = self.list_sensor_files()
        if len(files) <= keep_latest:
            return

        files_to_remove = files[:-keep_latest]
        for filename in files_to_remove:
            try:
                filepath = os.path.join(self.output, filename)
                os.remove(filepath)
                print(f"Removed old file: {filename}")
            except Exception as e:
                print(f"Error removing file {filename}: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, "_streaming_enabled") and self._streaming_enabled:
            self.stop_streaming()
