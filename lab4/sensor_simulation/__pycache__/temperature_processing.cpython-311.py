�
    ��g  �                   �   � d � Z d� ZdS )c                  �   � t           sdS t          t           �                    �   �         �  �        } | t          t           �  �        z  }|S )zR
    Calculates and returns the average temperature from the latest readings.
    N)�latest_temperatures�sum�values�len)�
total_temp�avg_temps     �dc:\Users\Admin\DSAI3202-Parallel-Distributed-Comp-3\lab4\sensor_simulation\temperature_processing.py�calculate_average_temperaturer
      sC   � � � ��t��(�/�/�1�1�2�2�J��C� 3�4�4�4�H��O�    c                  �   � t          d�  �         t          �                    �   �         D ]\  } }t          | � d|� d��  �         �dS )z]
    Prints the temperature report for all cities in the latest_temperatures dictionary.
    z
Temperature Report:z: u   °CN)�printr   �items)�city�temps     r	   �print_temperature_reportr      s^   � � 
�
!�"�"�"�)�/�/�1�1� $� $�
��d���"�"��"�"�"�#�#�#�#�$� $r   N)r
   r   � r   r	   �<module>r      s-   ��
� 
� 
�$� $� $� $� $r   