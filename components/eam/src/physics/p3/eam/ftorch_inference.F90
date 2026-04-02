module ftorch_inference

!   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real32

   ! get real kind from utils
!   use physics_utils, only: rtype,rtype8,btype

   use cam_logfile,  only: iulog
 
   ! Import our library for interfacing with PyTorch
   use ftorch, only : torch_model, torch_tensor, torch_kCPU, torch_delete, &
                      torch_tensor_from_array, torch_model_load, torch_model_forward

   implicit none

   ! Set working precision for reals
   integer, parameter :: wp = sp

   public ftorch_inference_cpu,init_ftorch_inference

   contains

   subroutine init_ftorch_inference(model_file_path,model)
        character(len=*), intent(in) :: model_file_path
        type(torch_model), intent(inout) :: model

        write(iulog, *) 'Torch model path:',  trim(model_file_path)
        write(iulog, *) 'torch_kCPU (device):',  torch_kCPU

   ! Load ML model
        call torch_model_load(model, trim(model_file_path), torch_kCPU) 
        write(iulog, *) 'Torch Model Loaded'   

   end subroutine init_ftorch_inference

   subroutine ftorch_inference_cpu(model,input_data,output_data)

   ! Pass in the model
        type(torch_model), intent(in) :: model

   ! ++ Change data strctures to match the new FTorch interface
   !! Set up Fortran data structures
   !     real(wp), dimension(:), intent(in) :: input_data
   !     real(wp), dimension(:), intent(inout) :: output_data

   ! Set up Fortran data structures
        real(wp), target, intent(in)    :: input_data(:)
        real(wp), target, intent(inout) :: output_data(:)

   ! Local aliases to satisfy newer FTorch interface
        real(wp), pointer, contiguous :: input_p(:)
        real(wp), pointer, contiguous :: output_p(:)
   !-- end change

   ! Set up Torch data structures
   ! a vector of input tensors (in this case we only have one), and the output tensor
 
        type(torch_tensor), dimension(1) :: in_tensors
        type(torch_tensor), dimension(1) :: out_tensors

   ! Test input
!        write(iulog, *) 'Inference Input:', input_data(:) 

!++ Change to match new FTorch interface
        input_p  => input_data
        output_p => output_data

   ! Create Torch input/output tensors from the above arrays
!        call torch_tensor_from_array(in_tensors(1), input_data, torch_kCPU)
!        call torch_tensor_from_array(out_tensors(1), output_data, torch_kCPU)
         call torch_tensor_from_array(in_tensors(1), input_p,  torch_kCPU)
         call torch_tensor_from_array(out_tensors(1), output_p, torch_kCPU)
!-- end change

   ! Inference
        call torch_model_forward(model, in_tensors, out_tensors)

   ! Print output for testing
   ! expected = [0.0_wp, 2.0_wp, 4.0_wp, 6.0_wp, 8.0_wp]  
   !    write(iulog, *) 'Inference Output:', output_data(:)

   ! Cleanup
        call torch_delete(in_tensors)
        call torch_delete(out_tensors)

   end subroutine ftorch_inference_cpu


end module ftorch_inference